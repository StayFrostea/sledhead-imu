# Contributing to Sled-Head IMU

## Development Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   make setup
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branching
- Create a branch per feature: `git checkout -b feature/your-feature-name`
- Use descriptive branch names that indicate the purpose

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose

### Testing
- Add tests for any new functions you add
- Run tests before submitting: `make test`
- Aim for good test coverage of new functionality

### Code Quality
Before submitting a PR, run:
```bash
make fmt && make lint && make test
```

## Pull Request Process

1. **Create a feature branch** from main
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Run quality checks**: `make fmt && make lint && make test`
5. **Commit your changes** with descriptive commit messages
6. **Push to your fork** and create a pull request
7. **Request review** from team members

## Commit Message Format

Use clear, descriptive commit messages:
- `feat: add new exposure calculation method`
- `fix: correct timestamp synchronization bug`
- `docs: update data dictionary`
- `test: add unit tests for filtering functions`

## Data Handling

- **Never commit large data files** to the repository
- Use Git LFS for large artifacts (see `.gitattributes`)
- Keep sample data files small and clearly marked as samples
- Document any data format changes in the data dictionary

## Notebook Guidelines

- Use the provided notebook templates in `notebooks/_templates/`
- Keep outputs clean and minimal
- Save important results to the appropriate `data/` subdirectory
- Use `nbstripout` to prevent committing large outputs

## Questions?

Feel free to ask questions in the team chat or create an issue for discussion.
