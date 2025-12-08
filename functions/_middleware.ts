// Middleware for Cloudflare Pages Functions
// Adds CORS headers and error handling

export async function onRequest(context: EventContext<any, any, any>) {
  // Add CORS headers
  const response = await context.next()

  response.headers.set('Access-Control-Allow-Origin', '*')
  response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')

  return response
}
