import { exec } from 'child_process';
import { promisify } from 'util';
import semver from 'semver';
import { logError } from './logger.js';

const execAsync = promisify(exec);

async function getPackageVersions(packageName) {
  try {
    const { stdout } = await execAsync(
      `npm show ${packageName} versions --json`,
    );

    const versions = JSON.parse(stdout);
    return versions;
  } catch (error) {
    logError(
      `Failed to fetch versions for ${packageName}. Details:`,
      error.message,
    );
    throw error;
  }
}

export async function getVulnerablePackageVersion(
  packageName,
  vulnerableVersionRange,
  firstPatchedVersion,
) {
  let versions = await getPackageVersions(packageName);

  let found_patch = false;
  let vulnerableVersionRangeList = vulnerableVersionRange
    .split(',')
    .map((v) => v.trim());

  // Iterate reverse-sorted versions to find the first vulnerable version
  for (let i = versions.length - 1; i >= 0; i--) {
    let version = versions[i];

    if (version == firstPatchedVersion) {
      found_patch = true;
    }

    if (found_patch) {
      let matched = false;

      for (let range of vulnerableVersionRangeList) {
        if (semver.satisfies(version, range)) {
          matched = true;
        } else {
          matched = false;
          break;
        }
      }

      if (matched) {
        return version;
      }
    }
  }

  return null;
}
