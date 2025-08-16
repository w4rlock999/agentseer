#!/bin/bash


# TODO, edit this bash script
# Exit script on error
set -e

# Variables (replace placeholders with actual values)
GIT_REPO_URL="<your-repo-url>"
NEW_PROJECT_NAME="<your-project-name>"
AUTHOR_NAME="<your-name>"
PROJECT_DESCRIPTION="<your-project-description>"

# Clone the repository
echo "Cloning repository from $GIT_REPO_URL..."
git clone $GIT_REPO_URL $NEW_PROJECT_NAME

# Navigate into the project directory
cd $NEW_PROJECT_NAME

# Remove existing Git configuration
echo "Removing existing Git configuration..."
rm -rf .git

# Reinitialize Git and set new remote
echo "Reinitializing Git..."
git init
git remote add origin $GIT_REPO_URL

# Update package.json
echo "Updating package.json..."
sed -i "s/\"name\": \".*\"/\"name\": \"$NEW_PROJECT_NAME\"/" package.json
sed -i "s/\"author\": \".*\"/\"author\": \"$AUTHOR_NAME\"/" package.json
sed -i "s/\"description\": \".*\"/\"description\": \"$PROJECT_DESCRIPTION\"/" package.json

# Update README.md
echo "Updating README.md..."
cat <<EOL > README.md
# $NEW_PROJECT_NAME

$PROJECT_DESCRIPTION

## Setup Instructions
1. Clone the repo: \`git clone $GIT_REPO_URL\`
2. Install dependencies: \`npm install\`
3. Start the development server: \`npm run dev\`
EOL

# Check if .env file exists
if [ -f ".env.example" ]; then
    echo "Copying .env.example to .env..."
    cp .env.example .env
else
    echo "Creating .env file..."
    touch .env
fi

# Update .gitignore
echo "Ensuring .env is in .gitignore..."
if ! grep -q ".env" .gitignore; then
    echo ".env" >> .gitignore
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Run the development server to test
echo "Testing the development server..."
npm run dev &

# Wait a bit and kill the server
sleep 5
kill $!

# Add changes to Git and push
echo "Committing changes to Git..."
git add .
git commit -m "Setup project with boilerplate"
git push -u origin main

echo "Project setup complete!"
