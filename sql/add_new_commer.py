import sqlite3


def add_car_entry(come, out, time):
    # Connect to the SQLite database
    conn = sqlite3.connect('../identifier.sqlite')
    cursor = conn.cursor()

    # Prepare the SQL query to insert a new row into the car table
    query = "INSERT INTO car (come, out, time) VALUES (?, ?, ?)"

    # Execute the SQL query with the provided values
    cursor.execute(query, (come, out, time))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()


# Example usage: Call the function with come=1, out=0, and current timestamp
come_value = 1
out_value = 0
current_time = '2024-05-30 12:00:00'  # Example timestamp
add_car_entry(come_value, out_value, current_time)
