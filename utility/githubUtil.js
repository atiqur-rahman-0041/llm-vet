import process from 'process';
import trim from 'lodash/trim.js';
import fetch from 'node-fetch';
import 'dotenv/config';
import {
  maxAdvisoryPerPage,
  defaultAdvirsoryPerPage,
  defaultSeverity,
} from '../constants.js';
import { logInfo, logError } from './logger.js';

const allowedSeverities = ['unknown', 'low', 'medium', 'high', 'critical'];

/*
 * Fetches advisories from the GitHub Advisory Database API.
 * @param {string} severity - The severity of the advisories to fetch. Possible values are 'unknown', 'low', 'medium', 'high', and 'critical'.
 * @param {number} n_advisories_per_page - The number of advisories to fetch per page. The maximum value is 100.
 */

export async function fetchAdvisories(
  severity = defaultSeverity,
  n_advisories_per_page = defaultAdvirsoryPerPage,
) {
  let lastPage = false;
  let advisories = [];
  let counter = 0;

  if (n_advisories_per_page > maxAdvisoryPerPage) {
    console.warn(
      'The maximum value for n_advisories_per_page is 100. Using 100 instead.',
    );
    n_advisories_per_page = maxAdvisoryPerPage;
  }

  if (!allowedSeverities.includes(severity)) {
    console.warn("Invalid severity value. Using 'high' instead.");
    severity = 'high';
  }

  let url = `https://api.github.com/advisories?ecosystem=npm&severity=${severity}&per_page=${n_advisories_per_page}`;

  while (!lastPage) {
    const headers = {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${process.env.GITHUB_ACCESS_TOKEN}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };

    try {
      let response = await fetch(url, { method: 'GET', headers });
      let data = await response.json();
      advisories = advisories.concat(data);

      let linkHeader = response.headers.get('Link');

      if (!linkHeader) {
        logInfo('Invalid link header. Stopping.');
        break;
      }

      [url, lastPage] = [
        trim(linkHeader.split(';')[0], '<> '),
        linkHeader.includes('rel="last"'),
      ];

      counter += 1;

      if (counter === 5) {
        break;
      }

      // Pause between requests
      await new Promise((resolve) => setTimeout(resolve, 1000));
    } catch (error) {
      logError('Error fetching advisories:', error);
      break;
    }
  }

  logInfo(
    `Total of ${advisories.length} advisories fetched with severity ${severity}`,
  );

  return advisories;
}

export async function getPatches(
  repoPath,
  vulnerableTag,
  patchedTag,
  version = true,
) {
  let url = `https://api.github.com/repos/${repoPath}/compare/v${vulnerableTag}...v${patchedTag}`;

  if (!version) {
    url = `https://api.github.com/repos/${repoPath}/compare/${vulnerableTag}...${patchedTag}`;
  }

  try {
    let response = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/vnd.github.v3+json',
        Authorization: `Bearer ${process.env.GITHUB_ACCESS_TOKEN}`,
      },
    });

    let data = await response.json();

    if (!response.ok && version) {
      logInfo(
        `Failed to fetch patch changes for ${repoPath} from v${vulnerableTag} to v${patchedTag}. Switching to non-version tags.`,
      );
      return await getPatches(repoPath, vulnerableTag, patchedTag, false);
    } else if (!response.ok && !version) {
      throw new Error(data.message);
    }

    let patchesInfo = {};

    data['files'].forEach((file) => {
      patchesInfo[file['filename']] = file['patch'];
    });

    return patchesInfo;
  } catch (error) {
    logError(
      `Error fetching patch changes for ${repoPath} from ${vulnerableTag} to ${patchedTag}. Details:`,
      error.message,
    );
    return {};
  }
}
