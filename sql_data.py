import sqlite3

def view_product_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    try:
        # Execute a query to select all records from the products table
        cursor.execute('SELECT * FROM products')
        
        # Fetch all rows from the query
        rows = cursor.fetchall()
        
        # Check if any rows were returned
        if rows:
            # Print the table headers
            print(f"{'ID':<5} {'Product Name':<20} {'Brand Name':<20} {'Size':<10} {'MRP':<10} {'Expiry Date':<15} {'Mfg Date':<15} {'Timestamp'}")
            print("-" * 100)
            
            # Print each row
            for row in rows:
                print(f"{row[0]:<5} {row[1]:<20} {row[2]:<20} {row[3]:<10} {row[4]:<10} {row[5]:<15} {row[6]:<15} {row[7]}")
        else:
            print("No records found in the products table.")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the database connection
        conn.close()

def view_fruit_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('fruit_detections.db')
    cursor = conn.cursor()
    
    try:
        # Execute a query to select all records from the fruit_detections table
        cursor.execute('SELECT * FROM fruit_detections')
        
        # Fetch all rows from the query
        rows = cursor.fetchall()
        
        # Check if any rows were returned
        if rows:
            # Print the table headers
            print(f"{'ID':<5} {'Fruit Name':<15} {'Condition':<10} {'Count':<10} {'Timestamp'}")
            print("-" * 60)
            
            # Print each row
            for row in rows:
                print(f"{row[0]:<5} {row[1]:<15} {row[2]:<10} {row[3]:<10} {row[4]}")
        else:
            print("No records found in the fruit_detections table.")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the database connection
        conn.close()

def view_predictions_database():
    # Connect to the SQLite database
    conn = sqlite3.connect('nos_count.db')
    cursor = conn.cursor()
    
    try:
        # Execute a query to select all records from the predictions table
        cursor.execute('SELECT * FROM predictions')
        
        # Fetch all rows from the query
        rows = cursor.fetchall()
        
        # Check if any rows were returned
        if rows:
            # Print the table headers
            print(f"{'ID':<5} {'Class Name':<20} {'Count':<10} {'Timestamp'}")
            print("-" * 50)
            
            # Print each row
            for row in rows:
                print(f"{row[0]:<5} {row[1]:<20} {row[2]:<10} {row[3]}")
        else:
            print("No records found in the predictions table.")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the database connection
        conn.close()

def main():
    # Ask user for which database to view
    print("Choose which database you want to view:")
    print("1. Product Database (products.db)")
    print("2. Fruit Detection Database (fruit_detections.db)")
    print("3. Predictions Database (nos_count.db)")

    choice = input("Enter the number of the database you want to view (1, 2, or 3): ")

    if choice == '1':
        view_product_database()
    elif choice == '2':
        view_fruit_database()
    elif choice == '3':
        view_predictions_database()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
