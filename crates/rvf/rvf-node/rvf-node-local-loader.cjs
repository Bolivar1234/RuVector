/* Local fork loader: prefer the source-built binary shipped in this tarball. */
const { join } = require('path')

const nativeByPlatform = {
  'darwin-arm64': 'rvf-node.darwin-arm64.node',
  'darwin-x64': 'rvf-node.darwin-x64.node',
  'linux-x64': 'rvf-node.linux-x64-gnu.node',
  'linux-arm64': 'rvf-node.linux-arm64-gnu.node',
  'win32-x64': 'rvf-node.win32-x64-msvc.node',
}

const key = `${process.platform}-${process.arch}`
const filename = nativeByPlatform[key]
if (!filename) {
  throw new Error(`The local rvf-node fork has no binary for ${key}`)
}

const native = require(join(__dirname, filename))
module.exports = native
// Keep the CJS loader compatible with ESM named imports used by the SDK and
// the AgentDB staging probe. Node's CJS lexer recognizes explicit properties;
// assigning the native object wholesale alone is not sufficient.
module.exports.RvfDatabase = native.RvfDatabase
