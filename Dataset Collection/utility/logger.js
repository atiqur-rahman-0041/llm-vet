import fs from 'fs';
import path from 'path';
import { logDirectory, dataDirectory } from '../constants.js';

// Ensure that the directory exists
fs.mkdirSync(logDirectory, { recursive: true });
fs.mkdirSync(dataDirectory, { recursive: true });

// Create write streams for each file
const output = fs.createWriteStream(path.resolve(dataDirectory, 'output.json'));
const error = fs.createWriteStream(path.resolve(logDirectory, 'error.log'), {
  flags: 'a',
});
const info = fs.createWriteStream(path.resolve(logDirectory, 'info.log'), {
  flags: 'a',
});

// Custom log functions
export function logOutput(...messages) {
  output.write(messages.join(' ') + '\n');
}

export function logError(...messages) {
  error.write(messages.join(' ') + '\n');
}

export function logInfo(...messages) {
  info.write(messages.join(' ') + '\n');
}
