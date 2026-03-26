TradeAlchemy 📈

TradeAlchemy is a full-stack web application designed to analyze stock market trends and predict future price movements. By combining real-time data gathering with deep learning, it provides users with actionable insights into market behavior.

Key Features

This application utilizes PyTorch and LSTM models to forecast stock prices based on historical data. It gathers up-to-date market information through real-time web scraping to feed into the analysis pipeline. The secure entry point to the application is a user-friendly login portal located exactly at index.html. Behind the scenes, a robust data management system efficiently stores and retrieves user data, historical stock prices, and model predictions, all presented on a dynamic interactive dashboard.

Tech Stack

The frontend is built using standard web technologies including HTML5, CSS3, and JavaScript, with index.html specifically serving as the main login page. The backend logic is powered by Python and the Flask framework. For database management, the project relies on PostgreSQL. The machine learning and data processing components are driven by PyTorch for the LSTM neural network architecture, alongside Python web scraping libraries.

Project Structure

The repository is organized into several key directories. The static folder contains CSS, JavaScript, and image files. The templates folder holds the HTML files, prominently featuring the index.html login screen. The models directory stores the PyTorch LSTM models and training scripts. Web scraping modules for market data are kept in the scrapers folder. The core Flask application logic resides in app.py, and Python dependencies are listed in requirements.txt.

Getting Started

To get a copy of the project up and running on your local machine, ensure you have Python, PostgreSQL, and Git installed.

First, clone the repository using Git and navigate into the TradeAlchemy directory.

Next, set up and activate a Python virtual environment. Install the necessary dependencies by running the standard pip install command against the requirements.txt file.

For the database configuration, create a PostgreSQL database for the project and update the connection URI in app.py or your .env file with your specific credentials.

Finally, run the application by executing app.py with Python. You can then access the platform by opening your web browser and navigating to your local host address, where you will be greeted by the initial login page.

Contributing & License
Contributions, issues, and feature requests are highly encouraged. Feel free to check the repository's issues page to get involved. This project is open-source and available under the MIT License.
