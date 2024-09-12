# Contrib's Guide

## 🌳 Branch Names

The branch name should be a one or two word max (separated with a -) name of the feature.

## ⛓️ Committing

Yes, I'm familiar with **Conventional Commits**. It's a specification for writing clear, consistent, and meaningful commit messages, typically used in version control systems like Git. The purpose of using Conventional Commits is to make commit history more readable, automatable, and helpful in determining release versions.

Here's a basic breakdown of the format:

### Structure:

A conventional commit message consists of:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Elements:

1. **`<type>`**: This defines the category of the change. Common types are:
   - `feat`: A new feature for the project.
   - `fix`: A bug fix.
   - `chore`: Maintenance tasks (like refactoring or config changes) that don't affect the code.
   - `docs`: Changes to documentation only.
   - `style`: Changes that do not affect the meaning of the code (e.g., formatting).
   - `refactor`: A code change that neither fixes a bug nor adds a feature.
   - `test`: Adding or updating tests.
   - `perf`: A code change that improves performance.
   - `build`: Changes that affect the build system or dependencies.
   - `ci`: Changes to CI configuration files and scripts.
2. **`[optional scope]`**: This specifies what part of the code is affected (e.g., `feat(auth): ...`). This part is optional but can be useful in larger projects.

3. **`<description>`**: A brief description of the change. This should be concise and in the imperative mood (e.g., "add", "fix", not "added", "fixed").

4. **Optional Body**: A more detailed explanation of the changes made, especially for complex commits.

5. **Optional Footer**: This is used to reference issues or breaking changes. For example:
   - `BREAKING CHANGE: describe the change that breaks backwards compatibility`
   - `Fixes #1234`: Links the commit to a specific issue.

### Example:

```bash
feat(auth): add OAuth2 login functionality

Added support for OAuth2 login flow, allowing users to authenticate using third-party providers.
```

### Benefits:

- **Automation**: Conventional commits enable automated versioning, changelogs, and semantic releases.
- **Consistency**: Teams follow a standard way of writing commit messages.
- **Readability**: The history of commits becomes easier to follow and review.
