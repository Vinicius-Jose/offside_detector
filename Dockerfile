
# Use Python Alpine for smaller image size
FROM python:3.10-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=soccer.settings

# Set work directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    jpeg-dev \
    zlib-dev \
    freetype-dev \
    lcms2-dev \
    openjpeg-dev \
    tiff-dev \
    tk-dev \
    tcl-dev \
    harfbuzz-dev \
    fribidi-dev \
    libimagequant-dev \
    libxcb-dev \
    libpng-dev

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry to not create virtual environment
RUN poetry config virtualenvs.create false

RUN poetry lock

# Install Python dependencies
RUN poetry install 

# Copy project files
COPY . .

# Create media directory for uploaded images
RUN mkdir -p /app/app/media/images

# Collect static files
RUN cd app && python manage.py collectstatic --noinput

# Create migrations and migrate database
RUN cd app && python manage.py makemigrations offside
RUN cd app && python manage.py makemigrations
RUN cd app && python manage.py migrate

# Expose port
EXPOSE 8000

# Change to app directory
WORKDIR /app/app

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
