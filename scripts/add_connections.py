""" File for add connections for Airflow """

from airflow.models import Connection
from airflow import settings


def add_connection(conn_id, conn_uri):
    """ Add connections for Airflow """
    print(f"Adding connection: {conn_id} {conn_uri}")
    conn = Connection(
        conn_id=conn_id,
        uri=conn_uri
    )
    session = settings.Session()
    session.add(conn)
    session.commit()


def main():
    pass


if __name__ == "__main__":
    main()
