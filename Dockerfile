# 1. Use the exact Python version that worked for you
FROM python:3.11.8-slim

# 2. Set the working directory inside the server
WORKDIR /app

# 3. Copy only the requirements first (this makes builds faster)
COPY soulsync-backend/requirements.txt .

# 4. Install the heavy ML libraries (HF has plenty of RAM for this)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your backend code
COPY soulsync-backend/ .

# 6. Hugging Face REQUIRES your app to run on port 7860
EXPOSE 7860

# 7. Start the Flask server with Gunicorn on the correct port
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]