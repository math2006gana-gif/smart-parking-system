"""
Database Configuration
IMPORTANT: Change 'your_password_here' to your actual MySQL password
"""

import mysql.connector
from mysql.connector import Error

# Database connection settings
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'your_password_here',  # ⚠️ CHANGE THIS
    'database': 'car_parking'
}


def get_db_connection():
    """
    Create and return MySQL database connection
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        raise e


def test_connection():
    """
    Test database connection
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"✅ Connected to MySQL Server version: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()