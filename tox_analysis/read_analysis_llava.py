import joblib,torch
import torch.nn.functional as F
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--no_image_path', type=str,
                        help='an integer for the accumulator')

    args = parser.parse_args()
    image_path = args.image_path
    no_image_path = args.no_image_path

    print('image path: {}'.format(image_path))
    print('no image path: {}'.format(no_image_path))

    data_image = joblib.load(image_path)
    data_no_image = joblib.load(no_image_path)

    #f = open('read_analysis_llava.log', 'w')
    cos_layer_avg = {}
    count = {}
    avg_start_layer = 0
    count_start = 0

    avg_end_layer = 0
    count_end = 0

    att_layer = {}
    sorry_logits_layer = {}

    divers_layer_hidden = {} #每一层的hidden state的语义方差
    for i in range(min(len(data_image),len(data_no_image))):
        max_logits_I = -10000
        max_layer = 0
        start_layer = -1
        end_layer = -1

        for key in data_image[i]['hidden_dict'].keys():
            if key > 1 and start_layer == -1:
                diff_ = F.softmax(data_no_image[i]['logits_dict'][key].to(torch.float32),dim=-1) - F.softmax(
                         data_no_image[i]['logits_dict'][key - 1].to(torch.float32),dim=-1)
                _, max_diff_id = torch.topk(diff_, 5)
                if 7371 in max_diff_id or 315 in max_diff_id:
                    start_layer = key

            if key > 1 and start_layer != -1:
                diff_ = F.softmax(data_no_image[i]['logits_dict'][key].to(torch.float32),dim=-1)
                _, max_diff_id = torch.topk(diff_, 1)
                if 7371 in max_diff_id or 315 in max_diff_id:
                    end_layer = key

            #try:
            sim = F.cosine_similarity(data_image[i]['hidden_dict'][key].to(torch.float32), data_no_image[i]['hidden_dict'][key].to(torch.float32), dim=0)
            if not key in cos_layer_avg:
                cos_layer_avg[key] = sim
                count[key] = 1
            else:
                cos_layer_avg[key] += sim
                count[key] += 1

            if key < 32:
                att_layer[key] = torch.mean(data_no_image[i]['att_dict'][key],dim=0)

        avg_start_layer += start_layer
        avg_end_layer += end_layer
        count_start += 1

    for k in cos_layer_avg.keys():
        if count[k] > 0:
            print('layer is {} sim is {}'.format(k,cos_layer_avg[k] / count[k]))

    print('avg start layer is {}'.format(avg_start_layer / count_start))
    print('avg end layer is {}'.format(avg_end_layer / count_start))

    print('att layer is {}'.format(att_layer))

