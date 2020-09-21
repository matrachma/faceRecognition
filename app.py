import time

import mysql.connector
import os
import schedule

from recognizer.face_cv import FaceCV
from tools import url_to_image


def job(face):
    db_conn = mysql.connector.connect(
        host=os.getenv('DB_HOST', "host.docker.internal"),
        user=os.getenv('DB_USER', "mysql"),
        password=os.getenv('DB_PASSWORD', "d0ck3r"),
        database=os.getenv('DB_NAME', "my_database")
    )

    my_cursor = db_conn.cursor()

    sql = "SELECT * FROM image_list WHERE image_result ='' OR image_result IS NULL "

    my_cursor.execute(sql)

    img_list = my_cursor.fetchall()

    for img_record in img_list:
        url = img_record[1]
        try:
            print("processing image: {}".format(url))
            image = url_to_image(url)
            result = face.detect_face(image)
        except (Exception, ValueError) as e:
            print("exception occurred: {}".format(e))
            result = "x"
        if result == "":
            result = "0"
        print("got result: ", result)

        sql_update = "UPDATE image_list SET image_result = %s WHERE image_id = %s"
        data = (result, img_record[0])

        my_cursor.execute(sql_update, data)

        db_conn.commit()

    db_conn.close()


def main():
    face = FaceCV(depth=16, width=8)
    run_at_time = os.getenv('RUN_AT_TIME', "00:27")
    schedule.every(10).minutes.do(job, face)
    schedule.every().day.at(run_at_time).do(job, face)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
