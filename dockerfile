# 1. Use official Node.js image
FROM node:20-alpine

# 2. Set working directory inside the container
WORKDIR /usr/src/app

# 3. Copy package.json and package-lock.json (if exists)
COPY package*.json ./

# 4. Install dependencies
RUN npm install

# 5. Copy the rest of your project files
COPY . .

# 6. Build Next.js app (creates the `.next` directory)
RUN npm run build

# 7. Expose port (Hugging Face and local both map this)
EXPOSE 3000

# 8. Start Next.js in production mode
CMD ["sh", "-c", "npm run start -- -p ${PORT:-3000}"]
