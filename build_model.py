import argparse

from tensorflow.keras.models import load_model

from config import KERAS_MODEL_FILE
from weights import WeightReader, download_yolov3_weights
from yolov3_model import make_modified_yolov3_model, make_yolov3_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modified", action="store_true")
    parser.add_argument("--weights", default="yolov3.weights")
    parser.add_argument("--output", default=str(KERAS_MODEL_FILE))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--load-existing", action="store_true")
    args = parser.parse_args()

    if args.load_existing:
        model = load_model(args.output, compile=False)
    else:
        if args.download:
            download_yolov3_weights(args.weights)
        model = make_modified_yolov3_model() if args.modified else make_yolov3_model()
        WeightReader(args.weights).load_weights(model)
        model.save(args.output)


if __name__ == "__main__":
    main()
