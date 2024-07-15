from detection import PersonDetection
from notifications import Notification
import os
from dotenv import load_dotenv

import argparse


def main(capture_index):

    #Load environment variables
    load_dotenv()

    sender_mail = os.environ.get("SENDER_MAIL")
    receiver_mail = os.environ.get("RECEIVER_MAIL")
    password = os.environ.get("PASSWORD")
    
    # Instanciate Notification and PersonDetection classes
    notification = Notification(sender_mail, receiver_mail, password)
    detection = PersonDetection(capture_index, notification)

    #Detection
    detection()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Lancement du sysytème de détection de personne")
    parser.add_argument('--capture_index', default=0, help="Index de l'adresse IP de la caméra à utiliser pour la capture")
    args = parser.parse_args()
main(args.capture_index)