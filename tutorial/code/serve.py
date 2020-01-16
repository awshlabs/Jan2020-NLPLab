import collections, json, logging, warnings
import multiprocessing as mp
from functools import partial

import gluonnlp as nlp
import mxnet as mx
from mxnet.gluon import Block, nn
import bert
from bert.data.qa import preprocess_dataset, SQuADTransform
import bert_qa_evaluate



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


def model_fn(model_dir = "", params_path = "bert_qa-7eb11865.params"):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a Gluon model, and the vocabulary
    """
    bert_model, vocab = nlp.model.get_model('bert_12_768_12',
                                        dataset_name='book_corpus_wiki_en_uncased',
                                        use_classifier=False,
                                        use_decoder=False,
                                        use_pooler=False,
                                        pretrained=False)
    net = BertForQA(bert_model)
    if len(model_dir) > 0:
        params_path = model_dir + "/" +params_path
    net.load_parameters(params_path, ctx=mx.cpu())
    
    tokenizer = nlp.data.BERTTokenizer(vocab,  lower=True)
    transform = SQuADTransform(tokenizer, is_pad=False, is_training=False, do_lookup=False)
    return net, vocab, transform


def transform_fn(model, input_data, input_content_type=None, output_content_type=None):
    """
    Transform a request using the Gluon model. Called once per request.
    :param model: The Gluon model and the vocab
    :param dataset: The request payload
    
        Example:
        ## (example_id, [question, content], ques_cont_token_types, valid_length, _, _)


        (2, 
        '56be4db0acb8001400a502ee', 
        'Where did Super Bowl 50 take place?', 
        
        'Super Bowl 50 was an American football game to determine the champion of the National 
        Football League (NFL) for the 2015 season. The American Football Conference (AFC) 
        champion Denver Broncos defeated the National Football Conference (NFC) champion 
        Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played 
        on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, 
        California. As this was the 50th Super Bowl, the league emphasized the "golden 
        anniversary" with various gold-themed initiatives, as well as temporarily suspending 
        the tradition of naming each Super Bowl game with Roman numerals (under which the 
        game would have been known as "Super Bowl L"), so that the logo could prominently 
        feature the Arabic numerals 50.', 
        
        ['Santa Clara, California', "Levi's Stadium", "Levi's Stadium 
        in the San Francisco Bay Area at Santa Clara, California."], 
        
        [403, 355, 355])

    :param input_content_type: The request content type, assume json
    :param output_content_type: The (desired) response content type, assume json
    :return: response payload and content type.
    """
    net, vocab, squadTransform = model
#     data = input_data
    data = json.loads(input_data)
#     test_examples_tuples = [(i, "", question, content, [], [])]
#     question, context = data #.split(" [CONTEXT] ")
#     tup = (0, "", question, context, [], [])
    test_examples_tuples = _test_example_transform(data)
    test_dataset = mx.gluon.data.SimpleDataset(test_examples_tuples)  # [tup]
    all_results = get_all_results(net, vocab, squadTransform, test_dataset, ctx=mx.cpu())
    all_predictions = collections.defaultdict(list) # collections.OrderedDict()
    data_transform = test_dataset.transform(squadTransform._transform)
    for features in data_transform:
        f_id = features[0].example_id
        results = all_results[f_id]
        prediction, nbest = bert_qa_evaluate.predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(vocab))        
        nbest_prediction = [] 
        for i in range(3):
            nbest_prediction.append('%.2f%% \t %s'%(nbest[i][1] * 100, nbest[i][0]))
        all_predictions[f_id] = nbest_prediction
    response_body = json.dumps(all_predictions)
    return response_body, output_content_type