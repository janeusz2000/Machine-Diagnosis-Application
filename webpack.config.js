const path = require('path');
const application = {
  mode: 'development',
  devtool: false,
  entry: './entry/application.js',
  output: {filename: 'application.js', path: path.resolve(__dirname, 'dist')},
};

module.exports = [application];