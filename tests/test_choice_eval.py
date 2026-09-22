"""Independent probability oracles for answer-only causal evaluation."""

import math
from types import SimpleNamespace

import pytest
import torch

from prophet.eval.choices import continuation_nats, encode_choice, rank_choices
from scripts.eval_arc_recovery import encoded_items, evaluate_items
from scripts.prepare_arc_recovery_eval import prepare_rows


class CharacterTokenizer:
    def encode(self, text, *, add_eos=False):
        assert add_eos is False
        return list(text.encode('ascii'))

    def byte_length(self, ids):
        return len(ids)


class FixedDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        probabilities = torch.full((128,), 0.2/126)
        probabilities[ord('a')] = 0.5
        probabilities[ord('b')] = 0.3
        self.register_buffer('logits', probabilities.log())

    def forward(self, tokens, *, return_mtp, loop_k=None):
        assert return_mtp is False
        return SimpleNamespace(logits=self.logits.expand(*tokens.shape, -1).clone())


def source_row(name='one'):
    return {'id': name, 'question': 'Which?', 'choices': {'label': ['1','2'], 'text': ['a','b']},
            'answerKey': '1'}


def test_causal_alignment_and_prompt_exclusion_match_markov_oracle():
    class Markov(torch.nn.Module):
        def forward(self, tokens, **kw):
            probabilities = torch.tensor([[0.1,0.6,0.2,0.1], [0.1,0.1,0.7,0.1],
                                          [0.1,0.1,0.1,0.7], [0.25,0.25,0.25,0.25]])
            return SimpleNamespace(logits=probabilities[tokens].log())
    # Context [0,1]; answer [2,3] has probabilities .7 and .7.
    # Including the prompt would wrongly add -log(.6).
    assert continuation_nats(Markov(), [0,1,2,3], 2) == pytest.approx(-math.log(.7*.7))
    assert continuation_nats(Markov(), [1,2], 1) == pytest.approx(-math.log(.7))


def test_complete_evaluation_with_variable_choice_counts():
    tok = CharacterTokenizer()
    rows = [source_row(), source_row('two')]
    rows[1]['choices'] = {'label':['A','B','C'], 'text':['b','a','aa']}
    rows[1]['answerKey'] = 'B'
    prepared = prepare_rows(rows, tok)
    result = evaluate_items(FixedDistribution(), encoded_items(prepared, tok, max_tokens=512))
    assert result['rows'] == 2 and result['accuracy'] == 1
    assert result['uniform_choice_chance'] == pytest.approx((1/2+1/3)/2)
    space = .2/126
    assert result['gold_answer_total_nats'] == pytest.approx(-2*math.log(space*.5))
    assert result['gold_answer_scored_tokens'] == result['gold_answer_scored_bytes'] == 4
    assert result['gold_answer_bits_per_byte'] == pytest.approx(-math.log2(space*.5)/2)
    assert result['items'][0]['prediction'] == 0 and result['items'][1]['prediction'] == 1


def test_character_normalization_can_reverse_ranking_and_ties_are_visible():
    result = rank_choices([.6, 1.0], [1,10])
    assert result['prediction'] == 0 and result['prediction_character_normalized'] == 1
    tie = rank_choices([1.,1.,3.], [2,2,1])
    assert tie == {'prediction':0,'prediction_character_normalized':0,'ties':2,'ties_character_normalized':2}


def test_no_boundary_retokenization_or_silent_truncation():
    class Merging(CharacterTokenizer):
        def encode(self, text, **kw):
            return [1] if text == 'x' else [2,3]
    with pytest.raises(ValueError, match='boundary'):
        encode_choice(Merging(), 'x', 'y', max_tokens=9)
    with pytest.raises(ValueError, match='truncation'):
        encode_choice(CharacterTokenizer(), 'hello', 'world', max_tokens=5)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_bad_losses_cannot_receive_a_rank(bad):
    with pytest.raises(ValueError, match='scores'):
        rank_choices([1.,bad], [1,2])


def test_nonfinite_or_reduced_precision_outputs_fail():
    model = FixedDistribution()
    model.logits[0] = float('-inf')
    with pytest.raises(ValueError, match='nonfinite'):
        continuation_nats(model, [1,2], 1)
    with pytest.raises(ValueError, match='FP32'):
        continuation_nats(FixedDistribution().half(), [1,2], 1)


def test_bad_labels_duplicate_rows_and_changed_tokens_fail():
    tok = CharacterTokenizer()
    row = source_row()
    with pytest.raises(ValueError, match='duplicate'):
        prepare_rows([row,row], tok)
    row['answerKey'] = 'C'
    with pytest.raises(ValueError, match='invalid'):
        prepare_rows([row], tok)
    items = prepare_rows([source_row()], tok)
    items[0]['tokens'][0]['input_ids_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='tokenization'):
        encoded_items(items, tok, max_tokens=512)


@pytest.mark.parametrize('start', [0,2,3])
def test_answer_boundary_requires_context_and_target(start):
    with pytest.raises(ValueError, match='preceding context'):
        continuation_nats(FixedDistribution(), [1,2], start)
