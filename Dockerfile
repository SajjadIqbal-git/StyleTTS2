# Use a base image with the necessary dependencies
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set a working directory
WORKDIR .

# Copy your project files
COPY . /STYLETTS

# Install any required packages (adjust to match your project needs)
RUN apt-get update && apt-get install -y \
    python3-pip \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port Gradio will run on (default: 7860)
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py"]
