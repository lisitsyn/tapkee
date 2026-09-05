#!/usr/bin/env node

/**
 * Script to render the HTML preview page as a PNG image for social media tags.
 * Uses Puppeteer to capture the visualization as tapkee-preview.png
 * Serves files locally to allow promoters.js to load the data
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const http = require('http');
const url = require('url');

// Simple static file server
function createServer() {
    return http.createServer((req, res) => {
        const parsedUrl = url.parse(req.url);
        let pathname = parsedUrl.pathname;
        
        // Serve the HTML file at root
        if (pathname === '/' || pathname === '/render_preview.html') {
            pathname = '/render_preview.html';
        }
        
        const filePath = path.join(__dirname, pathname);
        
        // Check if file exists
        if (!fs.existsSync(filePath)) {
            res.writeHead(404, {'Content-Type': 'text/plain'});
            res.end('File not found');
            return;
        }
        
        // Determine content type
        const ext = path.extname(filePath);
        const contentTypes = {
            '.html': 'text/html',
            '.js': 'text/javascript',
            '.json': 'application/json',
            '.css': 'text/css'
        };
        
        const contentType = contentTypes[ext] || 'text/plain';
        
        // Read and serve file
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500, {'Content-Type': 'text/plain'});
                res.end('Internal server error');
                return;
            }
            
            res.writeHead(200, {'Content-Type': contentType});
            res.end(data);
        });
    });
}

async function renderPreview() {
    let browser;
    let server;
    
    try {
        // Start local server
        console.log('Starting local server...');
        server = createServer();
        const port = 3000;
        server.listen(port);
        
        console.log('Starting browser...');
        browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const page = await browser.newPage();
        
        // Set viewport to match the og:image dimensions
        await page.setViewport({
            width: 1200,
            height: 630,
            deviceScaleFactor: 1
        });
        
        // Load the HTML file from local server
        const htmlUrl = `http://localhost:${port}/`;
        
        console.log('Loading HTML page...');
        await page.goto(htmlUrl, {
            waitUntil: 'networkidle0',
            timeout: 30000
        });
        
        // Wait for D3 to load and render the visualization
        console.log('Waiting for visualization to render...');
        
        // Debug: check what's in the page
        const pageContent = await page.evaluate(() => {
            return {
                hasD3: typeof d3 !== 'undefined',
                hasJQuery: typeof $ !== 'undefined',
                hasCbclPlot: !!document.querySelector('#cbclPlot'),
                hasSvg: !!document.querySelector('#cbclPlot svg'),
                circleCount: document.querySelectorAll('#cbclPlot svg circle').length,
                errors: window.errors || []
            };
        });
        
        console.log('Page state:', pageContent);
        
        // Listen to console messages
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
        
        await page.waitForFunction(() => {
            const svg = document.querySelector('#cbclPlot svg');
            const circles = document.querySelectorAll('#cbclPlot svg circle');
            console.log('Checking:', { svg: !!svg, circles: circles.length });
            return svg && circles.length > 0;
        }, { timeout: 15000 });
        
        // Additional wait to ensure everything is fully rendered
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        // Hover over a random datapoint to show tooltip
        console.log('Triggering hover on random datapoint...');
        await page.evaluate(() => {
            const circles = document.querySelectorAll('#cbclPlot svg circle');
            if (circles.length > 0) {
                // Pick a random circle (but not too close to edges for better visibility)
                const randomIndex = Math.floor(Math.random() * circles.length);
                const randomCircle = circles[randomIndex];
                
                // Create and dispatch mouseover event
                const mouseOverEvent = new MouseEvent('mouseover', {
                    view: window,
                    bubbles: true,
                    cancelable: true
                });
                randomCircle.dispatchEvent(mouseOverEvent);
                
                // Also trigger jQuery hover if needed
                if (typeof $ !== 'undefined') {
                    $(randomCircle).trigger('mouseenter');
                }
                
                console.log(`Hovered over circle ${randomIndex}`);
                return true;
            }
            return false;
        });
        
        // Wait for tooltip to appear and stabilize
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Ensure output directory exists
        const outputDir = path.resolve(__dirname, 'resources/public/img');
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        // Capture screenshot
        const outputPath = path.join(outputDir, 'tapkee-preview.png');
        console.log('Capturing screenshot...');
        
        await page.screenshot({
            path: outputPath,
            type: 'png'
        });
        
        console.log(`Preview image saved to: ${outputPath}`);
        
    } catch (error) {
        console.error('Error rendering preview:', error);
        process.exit(1);
    } finally {
        if (browser) {
            await browser.close();
        }
        if (server) {
            server.close();
        }
    }
}

// Check if puppeteer is available
try {
    require.resolve('puppeteer');
    renderPreview();
} catch (e) {
    console.error('Puppeteer is not installed. Please run: npm install puppeteer');
    console.log('Alternatively, you can install it globally: npm install -g puppeteer');
    process.exit(1);
}