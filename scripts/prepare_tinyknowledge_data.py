# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "typer",
#     "spacy",
#     "datasets",
#     "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
#     "tqdm",
#     "networkx",
# ]
# ///

import json
from collections import Counter
from collections import defaultdict

import networkx as nx
import spacy
import typer
from datasets import load_dataset
from tqdm import tqdm


def find_largest_independent_set(output_data: dict) -> list[str]:
  G = nx.Graph()
  nouns = list(output_data.keys())
  G.add_nodes_from(nouns)

  # Add edges between nouns that share adjectives (conflicts)
  for i, n1 in enumerate(nouns):
    adjs1 = set(output_data[n1].keys())
    for n2 in nouns[i + 1 :]:
      if adjs1 & set(output_data[n2].keys()):
        G.add_edge(n1, n2)

  # The maximum independent set of G is the maximum clique of its complement
  complement_G = nx.complement(G)
  cliques = nx.find_cliques(complement_G)
  return max(cliques, key=len, default=[])


def main(
  n_stories: int = 30000,
  min_frequency: int = 10,
) -> None:
  """
  Prepare a noun-adjective frequency dataset from TinyStories vocabulary.
  Extracts patterns like "The [noun] was [adjective]".
  """
  typer.echo('Loading Spacy model...')
  nlp = spacy.load('en_core_web_sm')

  typer.echo(f'Loading TinyStories dataset (scanning {n_stories} stories)...')
  dataset = load_dataset('roneneldan/TinyStories', split='train')

  # Dictionary: noun -> adjective -> count
  noun_adj_counts: dict[str, Counter] = defaultdict(Counter)

  typer.echo('Collecting canonical noun-adjective pairs from dataset...')
  for story in tqdm(dataset.take(n_stories), total=n_stories):
    doc = nlp(story['text'])
    for token in doc:
      # Look for AUX/VERB 'was' as the head, having 'nsubj' (NOUN) and 'acomp'/'attr' (ADJ)
      if (
        token.text.lower() == 'was'
        and token.lemma_ == 'be'
        and token.pos_ in ['AUX', 'VERB']
      ):
        nsubj = next(
          (c for c in token.children if c.dep_ == 'nsubj' and c.pos_ == 'NOUN'), None
        )
        acomp = next(
          (c for c in token.children if c.dep_ in ['acomp', 'attr'] and c.pos_ == 'ADJ'),
          None,
        )

        if nsubj and acomp:
          noun_lemma = nsubj.lemma_.lower()
          adj_lemma = acomp.lemma_.lower()

          # Basic validation to ensure purely alphabetical, meaningful words
          if (
            noun_lemma.isalpha()
            and adj_lemma.isalpha()
            and len(noun_lemma) > 1
            and len(adj_lemma) > 1
          ):
            noun_adj_counts[noun_lemma][adj_lemma] += 1

  # Convert to standard dict and filter by minimum frequency
  output_data = {}
  for noun, adjs in noun_adj_counts.items():
    filtered_adjs = {
      adj: count for adj, count in adjs.most_common() if count >= min_frequency
    }
    if filtered_adjs:
      output_data[noun] = filtered_adjs

  typer.echo(f'Found {len(output_data)} valid [noun] was [adj] pairings.')
  print(json.dumps(output_data, indent=2))

  independent_set = find_largest_independent_set(output_data)
  typer.echo(
    f'\nLargest set of {len(independent_set)} nouns without overlapping adjectives.'
  )
  print(json.dumps({noun: output_data[noun] for noun in independent_set}, indent=2))


if __name__ == '__main__':
  typer.run(main)
