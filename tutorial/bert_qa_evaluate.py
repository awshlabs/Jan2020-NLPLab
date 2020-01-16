# coding=utf-8

# Copyright 2018 The Google AI Language Team Authors, Allenai and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bert SQuAD evaluate."""
import logging, re, string
from collections import Counter, namedtuple, OrderedDict

import mxnet as mx
from mxnet import nd
import mxnet as mx
import gluonnlp as nlp
from mxnet.gluon import Block, loss, nn
from mxnet.gluon.loss import Loss
from mxnet.gluon.model_zoo import model_store

import bert_qa_evaluate
from bert.data.qa import SQuADTransform

PredResult = namedtuple('PredResult', ['start', 'end'])


class BertForQA(Block):
    """Model for SQuAD task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for QA task.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, prefix=None, params=None):
        super(BertForQA, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.span_classifier = nn.Dense(units=2, flatten=False)

    def forward(self, inputs, token_types, valid_length=None):  # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        bert_output = self.bert(inputs, token_types, valid_length)
        output = self.span_classifier(bert_output)
        return output
    
    
class BertForQALoss(Loss):
    """Loss for SQuAD task with BERT.

    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):  # pylint: disable=unused-argument
        super(BertForQALoss, self).__init__(
            weight=None, batch_axis=0, **kwargs)
        self.loss = loss.SoftmaxCELoss()

    def hybrid_forward(self, F, pred, label):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        pred : NDArray, shape (batch_size, seq_length, 2)
            BERTSquad forward output.
        label : list, length is 2, each shape is (batch_size,1)
            label[0] is the starting position of the answer,
            label[1] is the ending position of the answer.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size,)
        """
        pred = F.split(pred, axis=2, num_outputs=2)
        start_pred = pred[0].reshape((0, -3))
        start_label = label[0]
        end_pred = pred[1].reshape((0, -3))
        end_label = label[1]
        return (self.loss(start_pred, start_label) + self.loss(
            end_pred, end_label)) / 2
    
    
def download_qa_ckpt():
    model_store._model_sha1['bert_qa'] = '7eb11865ecac2a412457a7c8312d37a1456af7fc'
    result = model_store.get_model_file('bert_qa', root='.')
    print('Downloaded checkpoint to {}'.format(result))
    return result


def simple_predict(dataset, all_results, vocab):
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=True)
    transform = SQuADTransform(tokenizer, is_pad=False, is_training=False, do_lookup=False)
    dev_dataset = dataset.transform(transform._transform)
    all_predictions = OrderedDict()
    for features in dev_dataset:
        results = all_results[features[0].example_id]
    
        prediction, nbest = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=True))
    
        print('\nContext: %s\n'%(' '.join(features[0].doc_tokens)))
        question = features[0].input_ids.index('[SEP]')
        print('Question: %s\n'%(' '.join((features[0].input_ids[1:question]))))
        print('Top predictions: ')
        for i in range(3):
            print('%.2f%% \t %s'%(nbest[i][1] * 100, nbest[i][0]))
        print('')


def get_all_results(net, vocab, squadTransform, test_dataset, ctx = mx.cpu()):
    all_results = collections.defaultdict(list)
    
    def _vocab_lookup(example_id, subwords, type_ids, length, start, end):
        indices = vocab[subwords]
        return example_id, indices, type_ids, length, start, end
    
    dev_data_transform, _ = preprocess_dataset(test_dataset, squadTransform)
    dev_data_transform = dev_data_transform.transform(_vocab_lookup, lazy=False)
    dev_dataloader = mx.gluon.data.DataLoader(dev_data_transform, batch_size=1, shuffle=False)
    
    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        batch_size = inputs.shape[0]
        output = net(inputs.astype('float32').as_in_context(ctx),
                     token_types.astype('float32').as_in_context(ctx),
                     valid_length.astype('float32').as_in_context(ctx))
        pred_start, pred_end = mx.nd.split(output, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = pred_start.reshape(batch_size, -1).asnumpy()
        pred_end = pred_end.reshape(batch_size, -1).asnumpy()

        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results[example_id].append(bert_qa_evaluate.PredResult(start=start, end=end))
    return(all_results)


def _test_example_transform(test_examples):
    """
    Change test examples to a format like SQUAD data.
    Parameters
    ---------- 
    test_examples: a list of (question, context) tuple. 
        Example: [('Which NFL team represented the AFC at Super Bowl 50?',
                 'Super Bowl 50 was an American football game ......),
                  ('Where did Super Bowl 50 take place?',,
                 'Super Bowl 50 was ......),
                 ......]
    Returns
    ----------
    test_examples_tuples : a list of SQUAD tuples
    """
    test_examples_tuples = []
    i = 0
    for test in test_examples:
        question, context = test[0], test[1]  # test.split(" [CONTEXT] ")
        tup = (i, "", question, context, [], [])
        test_examples_tuples.append(tup)
        i += 1
    return(test_examples_tuples)


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i, _ in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, tokenizer):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tok_text = ' '.join(tokenizer(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for i in tok_ns_to_s_map.keys():
        tok_index = tok_ns_to_s_map[i]
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text



def predict(features,
            results,
            tokenizer,
            max_answer_length=64,
            null_score_diff_threshold=0.0,
            n_best_size=10,
            version_2=False):
    """Get prediction results.

    Parameters
    ----------
    features : list of SQuADFeature
        List of squad features for the example.
    results : list of data.qa.PredResult
        List of model predictions for span start and span end.
    tokenizer: callable
        Tokenizer function.
    max_answer_length: int, default 64
        Maximum length of the answer tokens.
    null_score_diff_threshold: float, default 0.0
        If null_score - best_non_null is greater than the threshold predict null.
    n_best_size: int, default 10
        The total number of n-best predictions.
    version_2: bool, default False
        If true, the SQuAD examples contain some that do not have an answer.

    Returns
    -------
    prediction: str
        The final prediction.
    nbest : list of (str, float)
        n-best predictions with their probabilities.
    """
    _PrelimPrediction = namedtuple('PrelimPrediction',
                                   ['feature_index', 'start_index', 'end_index',
                                    'pred_start', 'pred_end'])

    _NbestPrediction = namedtuple(
        'NbestPrediction', ['text', 'pred_start', 'pred_end'])

    prelim_predictions = []
    score_diff = None

    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_pred_start = 0  # the start logit at the slice with min null score
    null_pred_end = 0  # the end logit at the slice with min null score

    for features_id, (result, feature) in enumerate(zip(results, features)):
        start_indexes = _get_best_indexes(result.start, n_best_size)
        end_indexes = _get_best_indexes(result.end, n_best_size)
        if version_2:
            feature_null_score = result.start[0] + \
                result.end[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = features_id
                null_pred_start = result.start[0]
                null_pred_end = result.end[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=features_id,
                        start_index=start_index,
                        end_index=end_index,
                        pred_start=result.start[start_index],
                        pred_end=result.end[end_index]))

    if version_2:
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                pred_start=null_pred_start,
                pred_end=null_pred_end))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.pred_start + x.pred_end),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(
                pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = feature.doc_tokens[orig_doc_start:(
                orig_doc_end + 1)]
            tok_text = ' '.join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(' ##', '')
            tok_text = tok_text.replace('##', '')

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ''
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                pred_start=pred.pred_start,
                pred_end=pred.pred_end))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if version_2:
        if '' not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text='',
                    pred_start=null_pred_start,
                    pred_end=null_pred_end))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text='empty', pred_start=0.0, pred_end=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.pred_start + entry.pred_end)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = nd.softmax(nd.array(total_scores)).asnumpy()

    nbest_json = []

    for (i, entry) in enumerate(nbest):
        nbest_json.append((entry.text, float(probs[i])))

    if not version_2:
        prediction = nbest_json[0][0]
    else:
        # predict '' iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.pred_start - \
            best_non_null_entry.pred_end

        if score_diff > null_score_diff_threshold:
            prediction = ''
        else:
            prediction = best_non_null_entry.text

    prediction = nbest_json[0][0]
    return prediction, nbest_json


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate the F1 scores.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Calculate the EM scores.
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_F1_EM(dataset, predict_data):
    """Calculate the F1 and EM scores of the predicted results.
    Use only with the SQuAD1.1 dataset.

    Parameters
    ----------
    dataset_file: string
        Path to the data file.
    predict_data: dict
        All final predictions.

    Returns
    -------
    scores: dict
        F1 and EM scores.
    """
    f1 = exact_match = total = 0
    for record in dataset:
        total += 1
        if record[1] not in predict_data:
            message = 'Unanswered question ' + record[1] + \
                ' will receive score 0.'
            print(message)
            continue
        ground_truths = record[4]
        prediction = predict_data[record[1]]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    scores = {'exact_match': exact_match, 'f1': f1}

    return scores
