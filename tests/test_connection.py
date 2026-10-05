"""
Database connection test.

Skipped automatically when PostgreSQL is not reachable (for example on a
machine without the database), so the rest of the test suite still runs.
"""

import unittest


class ConnectionTest(unittest.TestCase):
    def test_can_reach_project_database(self):
        try:
            from sqlalchemy import text
            from src.database.connection import engine

            with engine.connect() as connection:
                database_name = connection.execute(text("SELECT current_database();")).scalar()
        except Exception as error:
            self.skipTest(f"PostgreSQL not reachable: {error}")

        print("Connected successfully! Database:", database_name)
        self.assertTrue(database_name)


if __name__ == "__main__":
    unittest.main()
