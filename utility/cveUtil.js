import process from 'process';
import fetch from 'node-fetch';
import { logInfo } from './logger.js';
import 'dotenv/config';

export async function getCveDescription(cveId) {
  const url = `https://services.nvd.nist.gov/rest/json/cves/2.0/?cveId=${cveId}`;

  const headers = {
    apiKey: process.env.NVD_API_KEY,
  };

  try {
    const response = await fetch(url, { method: 'GET', headers });
    const data = await response.json();

    const descriptions = data.vulnerabilities[0].cve.descriptions;

    for (const description of descriptions) {
      if (description.lang === 'en') {
        return description.value;
      }
    }

    logInfo(
      `No English description found for CVE ${cveId}. Returning empty string.`,
    );
    return '';
  } catch (error) {
    logInfo(
      `Failed to fetch CVE description for ${cveId}. Returning empty string. Details:`,
      error.message,
    );
    return '';
  }
}
