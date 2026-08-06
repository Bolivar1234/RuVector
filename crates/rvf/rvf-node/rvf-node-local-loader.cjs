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

module.exports = require(join(__dirname, filename))
