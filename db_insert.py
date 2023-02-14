
import psycopg2
from pytz import timezone 
from datetime import datetime

connection = psycopg2.connect(host='216.48.181.154', database='ckdr_t',port='5432',user='amani',password='amani123')
cursor=connection.cursor()

ack = False

def insert_db(memberid, status, batchId, membertype, timestamp, cid):
    query='''INSERT INTO "AttendancePipelines" (cid, "batchId", timestampp, "memberId", status, "memberType") VALUES (%s,%s,%s,%s,%s,%s);'''

    cursor.execute(query,(cid, batchId, timestamp, memberid, status, membertype ))
    connection.commit()

    ack = True

    return ack

    


# a = insert_db("memberid", "punchin", 1, "membertype",datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'),"test")
# print(a)