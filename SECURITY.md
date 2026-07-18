# Security policy

## Supported version

Security fixes are applied to the latest revision of the default branch. Older
commits and locally modified copies are not supported.

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability or suspected disclosure of
patient information.

1. Use GitHub's private vulnerability reporting or open a private draft Security
   Advisory for this repository.
2. If that option is unavailable, contact the repository owner using the institutional
   contact listed on the
   [Ichikawa Lab website](https://www.clg.niigata-u.ac.jp/~sichikawa/).
3. Include the affected file/version, impact, reproduction steps, and suggested
   remediation. Do not attach real medical images or identifying metadata.

We will acknowledge a valid report, investigate its scope, and coordinate a fix and
disclosure timeline with the reporter. Response times cannot be guaranteed because
this is academic research software.

## Protected health information

The repository must never contain DICOM/NIfTI images, subject identifiers,
identifying acquisition metadata, private source paths, per-case derived outputs, or
credentials. If such information is discovered:

- stop cloning, sharing, and processing the affected revision;
- report it privately using the process above;
- do not reproduce the information in an issue, pull request, screenshot, or log;
- treat removal from the current branch as insufficient, because the data may remain
  in Git history, forks, caches, or release archives;
- follow the institution's incident-response and ethics procedures before resuming
  distribution.

## Scope and safe operation

This code is designed for offline research on trusted systems. It is not hardened as
a network service and should not process untrusted files without isolation. DICOM
conversion and segmentation call third-party software; review their security notices
and licenses separately. Dependencies are locked with uv, but users remain
responsible for vulnerability monitoring and timely updates.
