import joblib,torch
import torch.nn.functional as F
import argparse

def get_dola(pre_layer,final_layer):
    softmax_mature_layer = F.softmax(final_layer,
                                     dim=-1)  # shape: (num_features)
    softmax_premature_layers = F.softmax(pre_layer,
                                         dim=-1)
    M = 0.5 * (softmax_mature_layer + softmax_premature_layers)

    log_softmax_mature_layer = F.log_softmax(final_layer,
                                             dim=-1)  # shape: (batch_size, num_features)
    log_softmax_premature_layers = F.log_softmax(pre_layer,
                                                 dim=-1)
    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(
        -1)  # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
        -1)  # shape: (num_premature_layers, batch_size)
    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

    # 6. Reduce the batchmean
    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)

    return js_divs


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
        # 获取最大值及其索引
        # if not 'sorry' in data_no_image[i]['response']:
        #     continue
        # if 'sorry' in data_no_image[i]['response']:
        #     continue
        max_logits_I = -10000
        max_layer = 0
        start_layer = -1
        end_layer = -1
        #print('keys are:')
        #print(data_image[i]['logits_dict'].keys())
        for key in data_image[i]['hidden_dict'].keys():
            # print('\nlayer is {}  ****************************************'.format(key), file=f)
            # max_val_im, max_idx_im = torch.topk(data_image[i]['logits_dict'][key].to(torch.float32), 5)
            # max_val_no_im, max_idx_no_im = torch.topk(data_no_image[i]['logits_dict'][key].to(torch.float32), 5)

            ### 选择增长量最大的层
            # if key > 1:
            #     logits_I = F.softmax(data_no_image[i]['logits_dict'][key].to(torch.float32),dim=-1)[7371] - F.softmax(
            #         data_no_image[i]['logits_dict'][key - 1].to(torch.float32),dim=-1)[7371] + \
            #                F.softmax(data_no_image[i]['logits_dict'][key].to(torch.float32), dim=-1)[315] - F.softmax(
            #         data_no_image[i]['logits_dict'][key - 1].to(torch.float32), dim=-1)[315]
            #     if logits_I > max_logits_I:
            #         max_logits_I = logits_I
            #         max_layer = key

            ### 选择开始增长在所有词里面最大的层，做为安全机制的开始层：
            if key > 1 and start_layer == -1:
                diff_ = F.softmax(data_no_image[i]['logits_dict'][key].to(torch.float32),dim=-1) - F.softmax(
                         data_no_image[i]['logits_dict'][key - 1].to(torch.float32),dim=-1)
                _, max_diff_id = torch.topk(diff_, 5)
                if 7371 in max_diff_id or 315 in max_diff_id:
                    start_layer = key

            ### 选择拒绝token已经开始排到top-1的层，做为安全机制的结束层：
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
            # except:
            #     continue

            #### 查看attention的分布：
            #print(data_no_image[i]['att_dict'][key].shape)
            if key < 32:
                att_layer[key] = torch.mean(data_no_image[i]['att_dict'][key],dim=0)
            # if key > 20:
            #     print('sim is {}'.format(sim), file=f)
            #     print('with image: ', file=f)
            #     #print('dola value is {}'.format(get_dola(data_image[i]['logits_dict'][key].to(torch.float32),data_image[i]['logits_dict'][32].to(torch.float32))))
            #     for j in range(3):
            #         print((max_idx_im[j],tokenizer.decode([max_idx_im[j]]),max_val_im[j]), end=", ",file=f)
            #     print('\nno image: ', file=f)
            #     #print('dola value is {}'.format(
            #     #    get_dola(data_no_image[i]['logits_dict'][key].to(torch.float32), data_no_image[i]['logits_dict'][32].to(torch.float32))), file=f)
            #     for j in range(3):
            #         print((max_idx_no_im[j],tokenizer.decode([max_idx_no_im[j]]),max_val_no_im[j]), end=", ",file=f)

        # print('\nmax layer is {}, value is {}'.format(max_layer,max_logits_I),file=f)
        # print('\nstart layer is {} '.format(start_layer), file=f)
        avg_start_layer += start_layer
        avg_end_layer += end_layer
        count_start += 1
        # print('\nfinal output:', file=f)
        # print('with image: {}'.format(data_image[i]['response']), file=f)
        # print('no image: {}'.format(data_no_image[i]['response']), file=f)
        # print('*************************************', file=f)

    for k in cos_layer_avg.keys():
        if count[k] > 0:
            print('layer is {} sim is {}'.format(k,cos_layer_avg[k] / count[k]))

    print('avg start layer is {}'.format(avg_start_layer / count_start))
    print('avg end layer is {}'.format(avg_end_layer / count_start))

    print('att layer is {}'.format(att_layer))

