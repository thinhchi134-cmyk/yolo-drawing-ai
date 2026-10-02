from ultralytics import YOLO


def main():

    model = YOLO("yolo26n.pt")

    model.train(
        data="data.yaml",

        epochs=100,
        imgsz=1024,
        batch=8,

        device=0,

        project="runs",
        name="yolo26n",

        patience=10,

        workers=0,

        pretrained=True,

        plots=True,
    )


if __name__ == "__main__":
    main()