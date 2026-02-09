# Start from verison of Python originally used to develop the code
FROM python:3.11

# Install depdendencies of the Burgers solver
RUN pip install \
    numpy==2.4.2 \
    scipy==1.17.0

# Copy source code and driver into the image
COPY src /home/burgers/src/
COPY drivers/smooth_traveling_wave_mintest.py /home/burgers/drivers/

WORKDIR /home/burgers/drivers
