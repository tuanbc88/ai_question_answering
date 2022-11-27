import os, sys, argparse, gc
import pandas as pd

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, DefaultDataCollator
import datasets

print("TF Version: ", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#########################################################
MODEL_BERT_BASE_MULTILINGUAL = 'bert-base-multilingual-cased'
MODEL_BERT_LARGE_MULTILINGUAL = 'bert-large-multilingual-cased'
MODEL_XLM_ROBERTA_BASE = 'xlm-roberta-base'
MODEL_XLM_ROBERTA_LARGE = 'xlm-roberta-large'

AVAILABLE_MODELS = [
    MODEL_BERT_BASE_MULTILINGUAL,
    MODEL_BERT_LARGE_MULTILINGUAL,
    MODEL_XLM_ROBERTA_BASE,
    MODEL_XLM_ROBERTA_LARGE,
]

def check_model(model: str):
    if model not in AVAILABLE_MODELS:
        print(f"Only support {AVAILABLE_MODELS}")
        exit(1)

#########################################################
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_BERT_BASE_MULTILINGUAL, help="Pretrained model bert")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--maxlen", type=int, default=512, help="Max sentence length")
    parser.add_argument("--stride", type=int, default=128, help="Stride value for window slide")
    parser.add_argument("--use_fast", type=bool, default=True, help="Tokenize sentence with fast bpe")

    return parser.parse_args()

#########################################################
def preprocess_dataset(ds, tokenizer, maxlen, stride):
    questions = [q.strip() for q in ds["question"]]
    inputs = tokenizer(
        questions,
        ds["context"],
        max_length=maxlen,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    # answers = ds["answer"]
    answer_starts = ds["answer_start"]
    answer_ends = ds["answer_end"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        # answer = answers[sample_idx]
        start_char = answer_starts[sample_idx]
        end_char = answer_ends[sample_idx]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def generate_dataset(file_name, tokenizer, data_collator, maxlen, stride, batch_size):
    df = pd.read_csv(file_name)
    df.drop("title", axis=1, inplace=True)
    ds = datasets.Dataset.from_dict(df)

    dataset =  ds.map(
        lambda x: preprocess_dataset(x, tokenizer, maxlen, stride),
        batched=True,
        remove_columns=ds.column_names,
    )

    return dataset.to_tf_dataset(
        columns=[
            "input_ids",
            "start_positions",
            "end_positions",
            "attention_mask",
            "token_type_ids",
        ],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=batch_size,
    )

#########################################################
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, saved_model_name: str):
        super().__init__()

        self.min_loss = sys.float_info.max
        self.min_val_loss = sys.float_info.max

        self.saved_model_name = saved_model_name

    def on_epoch_end(self, epoch, logs={}):
        min_loss = logs.get('loss')
        min_val_loss = logs.get('val_loss')

        if min_loss <= self.min_loss and min_val_loss <= self.min_val_loss:
            self.min_loss = min_loss
            self.min_val_loss = min_val_loss

            print("\nsave model at epoch {}".format(epoch+1))
            # self.model.save("models/{}.h5".format(self.saved_model_name))
            self.model.save("models/{}".format(self.saved_model_name), save_format='tf')
            
#########################################################
if __name__ == "__main__":
    args = get_arguments()
    model_name = args.model
    lr = args.lr
    batch_size = args.bs
    epochs = args.epochs
    maxlen = args.maxlen
    stride = args.stride
    use_fast = args.use_fast

    check_model(model_name)

    print("##############################")
    print("Model :", model_name)
    print("Learning Rate :", lr)
    print("Batch Size :", batch_size)
    print("Epochs :", epochs)
    print("Max Token Length :", maxlen)
    print("Stride :", stride)
    print("Use Fast :", use_fast)
    print("##############################")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    data_collator = DefaultDataCollator(return_tensors="tf")
    dataset_tr = generate_dataset("datasets/train.csv", tokenizer, data_collator, maxlen, stride, batch_size)
    dataset_val = generate_dataset("datasets/validate.csv", tokenizer, data_collator, maxlen, stride, batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    cb = myCallback(model_name.replace("/", "-"))
    earlyStopCB = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

    history = model.fit(
        dataset_tr,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=[cb, earlyStopCB],
    )

    hist = pd.DataFrame(history.history)
    hist.to_csv("{}_bs{}_lr{}.csv".format(model_name.replace("/", "-"), batch_size, lr))