-- schema.sql
-- Run this once in MySQL to set up the database and locations table.

CREATE DATABASE IF NOT EXISTS traffic_accident_db;
USE traffic_accident_db;

CREATE TABLE IF NOT EXISTS locations (
    location_id INT AUTO_INCREMENT PRIMARY KEY,
    location_name VARCHAR(255) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    formatted_address VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
