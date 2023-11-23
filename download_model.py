import argparse
import gdown


def download_model(google_drive_link, output_file):
    gdown.download(google_drive_link, output_file, quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from Google Drive.")
    parser.add_argument("-link", "--google_drive_link", default='https://drive.google.com/uc?id=1fIGZTfPbrtc2sMEBE_xLmnApWK47_J9J', type=str, help="Google Drive link to the model file")
    parser.add_argument("-output", "--output_file", default='MODEL.pth', type=str, help="Output file name")
    args = parser.parse_args()
    download_model(args.google_drive_link, args.output_file)
