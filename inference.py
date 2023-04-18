from regression import *

def inference(test_img_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
    model = get_model(backbone=arg.backbone, model_num=arg.model_num, finetune=False)
    os.chdir(os.path.join(os.getcwd(), ".."))
    model.load_state_dict(torch.load(arg.ckpt))
    model.to(device)
    model.eval()

    # load transform
    transform = get_transform(train=False)

    # load test images and predict
    result = []
    ctr = 0
    for img_name in os.listdir(test_img_folder):
        ctr += 1
        if ctr % 100 == 0:
            print(ctr)
        # prediction
        img_path = os.path.join(test_img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img, _ = transform(img, None)
        img = img.unsqueeze(0).to(device)
        predicted = model(img)
        predicted = predicted.cpu().detach().numpy()

        # save result
        carina = predicted[0, :2]
        ett = predicted[0, 2:4]
        dist = predicted[0, 4]
        dist1 = np.sqrt(np.sum((carina - ett)**2))
        result.append([img_name, carina[0], carina[1], ett[0], ett[0], dist, dist1])

        # Draw result
        img = img.cpu().detach().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        mean = np.array([MU, MU, MU])
        std = np.array([STD, STD, STD])
        img = std * img + mean
        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        r = 3
        draw.ellipse((carina[0]-r, carina[1]-r, carina[0]+r, carina[1]+r), fill='green')
        draw.ellipse((ett[0]-r, ett[1]-r, ett[0]+r, ett[1]+r), fill='blue')
        img_save_path = os.path.join("/home/ec2-user/data/ranzcr/inference", img_name)
        img.save(img_save_path)
    
    # save result
    df = pd.DataFrame(result, columns=["image_name", "carina_x", "carina_y", "ett_x", "ett_y", "dist", "dist1"])
    df.to_csv("inference.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--model_num', type=int, default=1, help='Model number')
    parser.add_argument('--backbone', type=str, default='resnet', help='Pretrained backbone model')

    arg = parser.parse_args()
    test_img_folder = "/home/ec2-user/data/ranzcr/downsized_norm"
    inference(test_img_folder)