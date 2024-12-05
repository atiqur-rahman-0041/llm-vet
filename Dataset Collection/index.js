import { severityLevels } from './constants.js';
import { getCveDescription } from './utility/cveUtil.js';
import { fetchAdvisories, getPatches } from './utility/githubUtil.js';
import { getVulnerablePackageVersion } from './utility/helper.js';
import { logOutput, logInfo, logError } from './utility/logger.js';

let advisories = [];

for (let severity of severityLevels) {
  let fetched_advisories = await fetchAdvisories(severity, 20);
  advisories = advisories.concat(fetched_advisories);
}

logInfo(`Total of ${advisories.length} advisories fetched`);

// drop entries that has no patched version
advisories = advisories.filter((advisory) =>
  advisory['vulnerabilities'].every(
    (vulnerability) => vulnerability['first_patched_version'] !== null,
  ),
);
logInfo(
  `Total of ${advisories.length} advisories remaining after filtering out advisories with no patched version`,
);

// add cve_description to each advisory
for (let advisory of advisories) {
  advisory['cve_description'] = await getCveDescription(advisory['cve_id']);
}

logInfo(`Added CVE descriptions to all advisories`);

let counter = 0;
// add patches to each vulnerability in each advisory
for (let advisory of advisories) {
  let repoPath = advisory['source_code_location'].replace(
    'https://github.com/',
    '',
  );

  if (!repoPath) {
    logError(`No repo path found for advisory ${advisory['ghsa_id']}`);
    advisory['invalid_patches'] = true;
    continue;
  }
  for (let vulnerability of advisory['vulnerabilities']) {
    if (vulnerability['package']['ecosystem'] !== 'npm') {
      vulnerability['patches'] = {};
      continue;
    }
    let packageName = vulnerability['package']['name'];
    let vulnerableVersionRange = vulnerability['vulnerable_version_range'];
    let firstPatchedVersion = vulnerability['first_patched_version'];

    let vulnerablePackageVersion = await getVulnerablePackageVersion(
      packageName,
      vulnerableVersionRange,
      firstPatchedVersion,
    );
    vulnerability['vulnerable_version'] = vulnerablePackageVersion;
    vulnerability['patches'] = await getPatches(
      repoPath,
      vulnerablePackageVersion,
      firstPatchedVersion,
    );
  }

  counter += 1;
  if (counter % 10 === 0) {
    logInfo(`Processed patches for ${counter} advisories`);
  }
}

for (let advisory of advisories) {
  try {
    let has_patches = false;
    for (let vulnerability of advisory['vulnerabilities']) {
      if (Object.keys(vulnerability['patches']).length !== 0) {
        has_patches = true;
        break;
      }
    }
    advisory['invalid_patches'] = !has_patches;
  } catch (error) {
    logError(`Error processing advisory ${advisory['ghsa_id']}: ${error}`);
    advisory['invalid_patches'] = true;
  }
}

// drop entries that has invalid patches
advisories = advisories.filter((advisory) => !advisory['invalid_patches']);
logInfo(
  `Total of ${advisories.length} advisories remaining after filtering out advisories with invalid patches`,
);

// write advisories to file
logOutput(JSON.stringify(advisories, null, 2));
