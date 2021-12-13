import cv2


def WebCam_Visualization(cam: cv2.VideoCapture):
    """Web_Cam で撮影した画像を表示する関数．WebCam の接続確認などに使用する．

        ```The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or
    Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and
    pkg-config, then re-run cmake or configure script``` が発生した場合

    ```pip uninstall opencv-python```, ```pip install opencv-contrib-python``` で再インストールすると解決できる．

    参考: [OpenCVエラー：関数が実装されていません](https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented/43531919)

        Args:
            cam (cv2.VideoCapture): [description]
    """

    while True:
        ret, frame = cam.read()
        if frame == None:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(cv2.getBuildInformation())

    # device_num = 0
    # cam = cv2.VideoCapture(device_num)

    # WebCam_Visualization(cam)
