# Use the base Python image
FROM python:3.8
# Set the working directory inside the container
WORKDIR /app
# Copy necessary files into the container
COPY . /app
# Install the relevant packages
RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
# Set the default command to run application app which contains all scripts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
