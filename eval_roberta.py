import tensorflow as tf

import xlrd
import json
import numpy as np
import modeling
import tokenization2

all_count=0
correct=0
results=[]

paths=["./data/data1.xlsx","./data/data2.xlsx"]

bert_config = modeling.BertConfig.from_json_file("./pretrained/config_tf.json")
tokenizer = tokenization2.RobertaTokenizer.from_pretrained(r"./pretrained")

graph=tf.Graph()
with graph.as_default():

    input_ids_placehold=tf.placeholder(shape=[None,64],dtype=tf.int32)
    input_mask_placehold=tf.placeholder(shape=[None,64],dtype=tf.int32)
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_placehold,
        input_mask=input_mask_placehold,
        token_type_ids=None,
        use_one_hot_embeddings=False)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [2], initializer=tf.zeros_initializer())

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    saver = tf.train.Saver()

with  tf.Session(graph=graph) as sess:
    model_file = tf.train.latest_checkpoint("./output/roberta/")
    saver.restore(sess, model_file)
    for path in paths:
        data = xlrd.open_workbook(path)
        table = data.sheets()[0]
        nrows = table.nrows
        for j in range(1, nrows):
            all_input_ids = None
            all_input_masks=None
            all_labels=None
            all_count += 1
            samples = []
            sentences = table.row_values(j)[6].split("|")
            for sentence in sentences:
                sample = []
                sample.append(sentence)
                sample.append(table.row_values(j)[4])
                samples.append(sample)
            for i in range(len(samples)):
                guid = "%s" % (i)
                text_a = samples[i][0].lower()
                text_b = samples[i][1].lower()
                label = str(int(table.row_values(j)[3] >= 0.5))

                pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
                pad_token_segment_id = 0

                label_map = {}
                for (i, label) in enumerate(["0","1"]):
                    label_map[label] = i
                inputs = tokenizer.encode_plus(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=64,
                )
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                attention_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = 64 - len(input_ids)
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                label_id = label_map[label]
                #print(input_ids)

                if all_input_ids is None:
                    all_input_ids=np.array(input_ids,dtype=np.int).reshape([1,64])
                else:
                    all_input_ids=np.append(all_input_ids,np.array(input_ids,dtype=np.int).reshape([1,64]),axis=0)

                if all_input_masks is None:
                    all_input_masks=np.array(attention_mask,dtype=np.int).reshape(1,64)
                else:
                    all_input_masks=np.append(all_input_masks,np.array(attention_mask,dtype=np.int).reshape(1,64),axis=0)

                if all_labels is None:
                    all_labels=np.array([label_id],dtype=np.int)
                else:
                    all_labels=np.append(all_labels,np.array([label_id],dtype=np.int),axis=0)
            #print(all_input_ids.shape)
            #print(all_input_masks.shape)
            #print(all_labels.shape)

            prob=sess.run(probabilities,feed_dict={input_ids_placehold:all_input_ids,input_mask_placehold:all_input_masks})
            prob=np.array(prob)
            #print(prob.shape)
            output = np.argmax(prob, axis=1)
            #print(output)

            correct += int(np.any(output == all_labels))
            results.append({"answer": table.row_values(j)[4], "ref": table.row_values(j)[6],
                            "logits": output.tolist(), "label": int(all_labels[0]),
                            "iscorrect": int(np.any(output == all_labels))})
            print(correct)
            print(all_count)
            print(correct / all_count)
    with open("./results/results_roberta.json", "w", encoding="utf8") as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
