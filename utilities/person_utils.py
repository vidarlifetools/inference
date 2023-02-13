
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch

COCO_PERSON_CLASS = 1

class peoples:
    def __init__(self, device = None, *args):
        self.people_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        self.people_detector.to(self.device)
        self.people_detector.eval()


    def detect(self, color, threshold=0.5):
        pil_image = Image.fromarray(color)  # Load the image
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )  # Defing PyTorch Transform
        transformed_img = transform(pil_image)  # Apply the transform to the image
        pred = self.people_detector(
            [transformed_img.to(self.device)]
        )  # Pass the image to the model
        pred_classes = pred[0]["labels"].cpu().numpy()
        pred_boxes = [
            [i[0], i[1], i[2], i[3]]
            for i in list(pred[0]["boxes"].cpu().detach().numpy().astype(int))
        ]  # Bounding boxes
        pred_scores = list(pred[0]["scores"].cpu().detach().numpy())

        person_boxes = []
        # Select box has score larger than threshold and is person
        for pred_class, pred_box, pred_score in zip(
                pred_classes, pred_boxes, pred_scores
        ):
            if (pred_score > threshold) and (pred_class == COCO_PERSON_CLASS):
                person_boxes.append(pred_box)

        return person_boxes
