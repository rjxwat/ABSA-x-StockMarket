# Import necessary libraries
from bs4 import BeautifulSoup  # For parsing and extracting HTML content
from urllib import request     # For opening URLs
from selenium import webdriver  # For automating the web browser
from selenium.webdriver.common.by import By  # For locating elements on the webpage
from selenium.webdriver.support.ui import WebDriverWait  # For waiting until an element is available
from selenium.webdriver.support import expected_conditions as EC  # Conditions to wait for elements to appear
import pandas as pd  # For handling and saving data
from datetime import datetime  # For handling date formatting
import json  # For encoding headlines in JSON format

# Base URL for Economic Times Archive
base_url = "https://economictimes.indiatimes.com"

# Function to extract news headlines from a specific news archive link
def extract_news_text(link):
    news_list = []  # List to store the extracted headlines
    try:
        # Open the URL using urllib and parse the HTML using BeautifulSoup
        page = request.urlopen(link)
        soup = BeautifulSoup(page, "lxml")
        
        # Locate the main content section that holds the news
        content = soup.find_all("section", class_="main_container pd0")
        
        if content:
            # Find the section where the news headlines are located
            news = content[0].find_all("section", id="pageContent")
            if news:
                # Extract all the links (news headlines) and append to the list
                news = news[0].find_all("a")
                for n in news:
                    news_list.append(n.text.strip())  # Clean and add the headline to the list
    except Exception as e:
        # Print an error message if something goes wrong
        print(f"Failed to extract news from {link}: {e}")
    
    return news_list  # Return the list of extracted headlines

# Function to extract news for a specific month and year
def extract_news(month, year, max_days=3):
    driver = webdriver.Chrome()  # Open a Chrome browser instance using Selenium
    news_by_date = {}  # Dictionary to store the headlines by date
    
    # Construct the URL for the given month
    url = base_url + month
    driver.get(url)  # Open the URL in the browser
    wait = WebDriverWait(driver, 10)  # Set an explicit wait to ensure page elements load
    
    try:
        # Wait until the calendar element is located on the webpage
        calendar = wait.until(EC.presence_of_element_located((By.ID, "calender")))
        
        # Find all the links (dates) in the calendar
        links = calendar.find_elements(By.TAG_NAME, "a")
        
        for link in links:
            # Extract the date text from the link
            date = link.text.strip()
            if not date:
                continue  # If the date is empty, skip to the next link
            
            # Split and extract the month number from the month URL
            month_num = month.split(',')[1].split('-')[1].split('.')[0]
            full_date = f"{date} {month_num} {year}"  # Construct a full date string (dd mm yyyy)
            
            # Format the date into the desired "YYYY-MM-DD" format
            formatted_date = datetime.strptime(full_date, "%d %m %Y").strftime("%Y-%m-%d")
            
            # Get the URL for the news on that particular day
            news_url = link.get_attribute("href")
            
            # Extract headlines from the news page using the extract_news_text function
            headlines = extract_news_text(news_url)
            
            # Store the headlines in the dictionary using the date as the key
            news_by_date[formatted_date] = headlines
            
            # Print a message indicating that the headlines for the date have been processed
            print(f"Finished processing {formatted_date}")
    
    except Exception as e:
        # If there's an error, print an error message
        print(f"Failed to process month {month}: {e}")
    
    finally:
        driver.quit()  # Close the browser after processing is complete
    
    return news_by_date  # Return the dictionary containing dates and their corresponding headlines

# Function to generate a list of month URLs for a given year
def get_month_urls(year):
    # Generate the URL format for each month of the given year
    months = [f'/archive/year-{year},month-{i}.cms' for i in range(1, 13)]  # URL for 12 months (Jan to Dec)
    return months

# List of years for which we want to extract news
years = [2023, 2022, 2021, 2020, 2019, 2018]

# Dictionary to store all the extracted news data
all_news_data = {}

# Loop through each year and each month to extract the news
for year in years:
    # Get the URLs for all the months in the current year
    monthly_urls = get_month_urls(year)
    for month_url in monthly_urls:
        print(f"Processing month: {month_url}")  # Print a message indicating which month is being processed
        
        # Extract news for the current month and year
        news_data = extract_news(month_url, year)
        
        # Update the overall dictionary with the extracted news
        all_news_data.update(news_data)

# Convert the news data into a Pandas DataFrame
df = pd.DataFrame([(date, json.dumps(headlines)) for date, headlines in all_news_data.items()],
                  columns=['Date', 'Headlines'])  # Columns: Date, Headlines (stored as JSON string)

# Save the DataFrame to a CSV file for the last year in the list
df.to_csv(f"economic_times_news_{year}.csv", index=False)
print(f"Data extraction complete. CSV file created for the year {year}.")

# Print the first few rows of the DataFrame to verify the extracted data
print(df.head())
print(f"Total rows: {len(df)}")

# If the DataFrame is not empty, print a sample of headlines for the first date
if not df.empty:
    first_date = df.iloc[0]['Date']  # Get the date of the first row
    first_date_headlines = json.loads(df.iloc[0]['Headlines'])  # Parse the JSON string back into a list
    print(f"\nSample headlines for {first_date}:")
    
    # Print the first 5 headlines for the first date
    for headline in first_date_headlines[:5]:
        print(f"- {headline}")
    
    print("...")
else:
    print("No data was extracted.")  # Print a message if no data was extracted
