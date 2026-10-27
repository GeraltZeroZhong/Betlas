# Publishing a Betlas release

Betlas distributes the Python package through PyPI and the three application
checkpoints through GitHub release attachments. Publish the matching GitHub
release first so package users can install checkpoints on their first prediction.

## Build and check

Update the package version in `pyproject.toml`, the changelog, versioned
documentation links and checkpoint manifest for the release. Build from the
reviewed source tree with an empty output directory. For v1.1.0:

```bash
python -m pip install --upgrade build twine 'readme-renderer[md]'
python -m build --outdir dist/1.1.0
python -m twine check --strict \
  dist/1.1.0/betlas-1.1.0-py3-none-any.whl \
  dist/1.1.0/betlas-1.1.0.tar.gz
```

Inspect both archives for the expected modules, configuration, example and
checkpoint manifest. Preview the README with the Markdown renderer and check its
links. PyPI descriptions use absolute links to documentation at the release tag.

Install the wheel in a fresh environment, then run the README Quickstart and
the three prediction interfaces with the matching local checkpoints. Check that
the installed version is 1.1.0 and that imports resolve to the installed package.
The default build command creates the wheel from the source distribution,
checking that the source archive contains the required build inputs.

## Publish checkpoint attachments

Create the `v1.1.0` GitHub release from the reviewed source revision. Attach:

- `betlas-solenoid-v1.1.0.json`
- `betlas-barrel-v1.1.0.json`
- `betlas-barrel-strands-v1.1.0.json`
- `MODEL_CARDS.md`

The three checkpoint filenames, sizes and SHA-256 values must match
`src/betlas/prediction/manifest.json`. Verify installation through the default
release URLs in a fresh cache:

```bash
betlas checkpoints install all --cache-dir release-checkpoints
```

Check the release-tag documentation links before publishing the Python package.

## Upload to PyPI

Check PyPI for a version conflict, then upload the two validated
files. Authenticate with the project's authorised PyPI publishing credentials
through Twine's supported local authentication or its interactive prompt.

```bash
python -m twine upload --repository-url https://upload.pypi.org/legacy/ \
  dist/1.1.0/betlas-1.1.0-py3-none-any.whl \
  dist/1.1.0/betlas-1.1.0.tar.gz
```

Install the published version in a fresh environment:

```bash
python -m pip install --index-url https://pypi.org/simple 'betlas==1.1.0'
python -c "from importlib.metadata import version; print(version('betlas'))"
betlas --help
```

Repeat the Quickstart and checkpoint download check against that installation.
Review the rendered package page, download files and project links.
