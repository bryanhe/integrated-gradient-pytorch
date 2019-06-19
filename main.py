import numpy as np
import torch
import torchvision
from torchvision import models
import cv2
import torch.nn.functional as F
from utils import calculate_outputs_and_gradients, generate_entrie_images
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
import argparse
import os

parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='01.jpg', help='the images name')

if __name__ == '__main__':
    args = parser.parse_args()
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # start to create models...
    if args.model_type == 'inception':
        model = models.inception_v3(pretrained=True)
    elif args.model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    model = torchvision.models.densenet121(num_classes=250)
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to("cuda")
    # checkpoint = torch.load(args.root + args.patient + "_model.pt")
    checkpoint = torch.load("../histology/output/densenet121_224/top_250/" + "BC23209" + "_model.pt")
    model.load_state_dict(checkpoint["model"])
    for param in model.parameters():
        param.requires_grad = False


    model.eval()
    if args.cuda:
        model.cuda()
    # read the image
    img = cv2.imread('examples/' + args.img)
    if args.model_type == 'inception':
        # the input image's size is different
        img = cv2.resize(img, (299, 299))
    img = img.astype(np.float32) 
    img = img[:, :, (2, 1, 0)]
    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, args.cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=100, num_random_trials=25, cuda=args.cuda)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                        img_integrated_gradient_overlay)
    cv2.imwrite('results/' + args.model_type + '/' + os.path.splitext(args.img)[0] + "_img.jpg", np.uint8(img)[:, :, (2, 1, 0)])
    cv2.imwrite('results/' + args.model_type + '/' + os.path.splitext(args.img)[0] + "_exp.jpg", np.uint8(img_integrated_gradient[:, :, (2, 1, 0)]))
    cv2.imwrite('results/' + args.model_type + '/' + args.img, np.uint8(output_img))

    print(np.uint8(np.max(img_integrated_gradient, 2)))
    attribution = np.max(img_integrated_gradient, 2)
    attribution = np.uint8(cv2.GaussianBlur(attribution, (15, 15), 0))
    print(attribution.max())
    attribution = attribution / attribution.max()
    attribution *= 255
    attribution = np.uint8(attribution)
    heatmap = cv2.applyColorMap(attribution, cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    heatmap = np.float32(heatmap) / heatmap.max()
    vis = heatmap + np.mean(np.float32(img[:, :, (2, 1, 0)]) / 255, 2, keepdims=True)
    vis = vis / np.max(vis)
    cv2.imwrite('results/' + args.model_type + '/' + os.path.splitext(args.img)[0] + "_vis.jpg", np.uint8(255 * vis))
    cv2.imwrite('results/' + args.model_type + '/' + os.path.splitext(args.img)[0] + "_heatmap.jpg", np.uint8(255 * heatmap))
