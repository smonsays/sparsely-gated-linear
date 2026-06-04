# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "pandas",
#     "pyarrow",
# ]
# ///
import dataclasses
import os
import tempfile
import webbrowser
from pathlib import Path

import pandas as pd
import typer


@dataclasses.dataclass
class InteractiveTable:
  df: pd.DataFrame
  name: str
  search_pane_cols: list[str] = dataclasses.field(default_factory=list)

  def to_html(self) -> str:
    """Export pandas DataFrame to HTML using DataTables.js"""

    table_html = self.df.to_html(
      table_id=self.name, classes=['display'], escape=False, index=False
    )

    # We need to map column names to their indices for DataTables columnDefs
    cols = list(self.df.columns)

    search_pane_targets = []
    no_search_pane_targets = []

    for i, col in enumerate(cols):
      if col in self.search_pane_cols:
        search_pane_targets.append(i)
      else:
        no_search_pane_targets.append(i)

    col_defs = f"""[
      {{ searchPanes: {{ show: true }}, targets: {search_pane_targets} }},
      {{ searchPanes: {{ show: false }}, targets: {no_search_pane_targets} }}
    ]"""

    html_template = f"""
      <!DOCTYPE html>
      <html>
      <head>
        <title>DataTable</title>
        <link href="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-2.3.4/cc-1.1.0/b-3.2.0/sp-2.3.3/sl-2.1.0/datatables.min.css" rel="stylesheet" />
        <script src="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-2.3.4/cc-1.1.0/b-3.2.0/sp-2.3.3/sl-2.1.0/datatables.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 20px;
            }}
            .dtsp-searchPane {{
                margin-top: 10px;
                margin-bottom: 20px;
            }}
        </style>
      </head>
      <body>
        {table_html}
        <script>
          $(document).ready(function() {{
            $('#{self.name}').DataTable({{
              layout: {{
                  top1: 'searchPanes'
              }},
              columnDefs: {col_defs},
              pageLength: 25
            }});
          }});
        </script>
      </body>
      </html>
      """  # noqa: E501

    return html_template

  def write_html(self, filepath: str) -> None:
    with open(filepath, 'w') as f:
      f.write(self.to_html())

  def show(self) -> None:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
      tmp_file.write(self.to_html())
      tmp_filepath = tmp_file.name

    abs_path = os.path.abspath(tmp_filepath)
    webbrowser.open(f'file://{abs_path}')


def main(
  input_path: str = 'logs/20260313_141205_della-i20g2_5cbc_tinystories/gating_neighbors.parquet',
) -> None:
  """
  Reads a gating_neighbors.parquet file and creates an interactive HTML table.
  """
  typer.echo(f'Reading {input_path}...')
  df = pd.read_parquet(input_path)

  # Preprocess the data
  df['input_excerpt'] = (
    df['input_excerpt']
    .str.replace('Ġ', ' ', regex=False)
    .str.replace('Ċ', '\\n', regex=False)
  )
  df['input_excerpt'] = df['input_excerpt'].str.replace(
    r'\*\*(.*?)\*\*', r'<strong>\1</strong>', regex=True
  )
  df['top_k_predictions'] = (
    df['top_k_predictions']
    .str.join(' | ')
    .str.replace('Ġ', '', regex=False)
    .str.replace('Ċ', '\\n', regex=False)
  )

  # Ensure table name is a valid HTML id
  p = Path(input_path)
  table_name = p.stem.replace('-', '_').replace('.', '_')

  table = InteractiveTable(
    df=df, name=table_name, search_pane_cols=['layer', 'channel', 'position']
  )

  typer.echo('Opening HTML table in the browser...')
  table.show()


if __name__ == '__main__':
  typer.run(main)
