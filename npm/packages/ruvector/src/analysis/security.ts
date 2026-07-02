/**
 * Security Analysis Module - Consolidated security scanning
 *
 * Single source of truth for security patterns and vulnerability detection.
 * Used by native-worker.ts and parallel-workers.ts
 */

import * as fs from 'fs';

export interface SecurityPattern {
  pattern: RegExp;
  rule: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  suggestion?: string;
}

export interface SecurityFinding {
  file: string;
  line: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  rule: string;
  message: string;
  match?: string;
  suggestion?: string;
}

/**
 * Default security patterns for vulnerability detection
 */
export const SECURITY_PATTERNS: SecurityPattern[] = [
  // Critical: Hardcoded secrets
  { pattern: /password\s*=\s*['"][^'"]+['"]/gi, rule: 'no-hardcoded-password', severity: 'critical', message: 'Hardcoded password detected', suggestion: 'Use environment variables or secret management' },
  { pattern: /api[_-]?key\s*=\s*['"][^'"]+['"]/gi, rule: 'no-hardcoded-apikey', severity: 'critical', message: 'Hardcoded API key detected', suggestion: 'Use environment variables' },
  { pattern: /secret\s*=\s*['"][^'"]+['"]/gi, rule: 'no-hardcoded-secret', severity: 'critical', message: 'Hardcoded secret detected', suggestion: 'Use environment variables or secret management' },
  { pattern: /private[_-]?key\s*=\s*['"][^'"]+['"]/gi, rule: 'no-hardcoded-private-key', severity: 'critical', message: 'Hardcoded private key detected', suggestion: 'Use secure key management' },

  // High: Code execution risks
  { pattern: /eval\s*\(/g, rule: 'no-eval', severity: 'high', message: 'Avoid eval() - code injection risk', suggestion: 'Use safer alternatives like JSON.parse()' },
  { pattern: /exec\s*\(/g, rule: 'no-exec', severity: 'high', message: 'Avoid exec() - command injection risk', suggestion: 'Use execFile or spawn with args array' },
  { pattern: /Function\s*\(/g, rule: 'no-function-constructor', severity: 'high', message: 'Avoid Function constructor - code injection risk' },
  { pattern: /child_process.*exec\(/g, rule: 'no-shell-exec', severity: 'high', message: 'Shell execution detected', suggestion: 'Use execFile or spawn instead' },

  // High: SQL injection
  { pattern: /SELECT\s+.*\s+FROM.*\+/gi, rule: 'sql-injection-risk', severity: 'high', message: 'Potential SQL injection - string concatenation in query', suggestion: 'Use parameterized queries' },
  { pattern: /`SELECT.*\$\{/gi, rule: 'sql-injection-template', severity: 'high', message: 'Template literal in SQL query', suggestion: 'Use parameterized queries' },

  // Medium: XSS risks
  { pattern: /dangerouslySetInnerHTML/g, rule: 'xss-risk', severity: 'medium', message: 'XSS risk: dangerouslySetInnerHTML', suggestion: 'Sanitize content before rendering' },
  { pattern: /innerHTML\s*=/g, rule: 'no-inner-html', severity: 'medium', message: 'Avoid innerHTML - XSS risk', suggestion: 'Use textContent or sanitize content' },
  { pattern: /document\.write\s*\(/g, rule: 'no-document-write', severity: 'medium', message: 'Avoid document.write - XSS risk' },

  // Medium: Other risks
  { pattern: /\$\{.*\}/g, rule: 'template-injection', severity: 'low', message: 'Template literal detected - verify no injection' },
  { pattern: /new\s+RegExp\s*\([^)]*\+/g, rule: 'regex-injection', severity: 'medium', message: 'Dynamic RegExp - potential ReDoS risk', suggestion: 'Validate/sanitize regex input' },
  { pattern: /\.on\s*\(\s*['"]error['"]/g, rule: 'unhandled-error', severity: 'low', message: 'Error handler detected - verify proper error handling' },
];

/**
 * Build a table of line-start offsets for a file (offsets[i] = index of the
 * first character of line i+1). Computed once per file so per-match line
 * numbers are O(log n) instead of an O(n) slice+split per match (ADR-275).
 */
function buildLineOffsets(content: string): number[] {
  const offsets = [0];
  for (let i = 0; i < content.length; i++) {
    if (content.charCodeAt(i) === 10 /* '\n' */) {
      offsets.push(i + 1);
    }
  }
  return offsets;
}

/**
 * Binary-search the offset table for the 1-based line number containing
 * the given character index.
 */
function lineNumberAt(offsets: number[], index: number): number {
  let lo = 0;
  let hi = offsets.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= index) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo + 1;
}

/**
 * Scan a single file for security issues
 */
export function scanFile(
  filePath: string,
  content?: string,
  patterns: SecurityPattern[] = SECURITY_PATTERNS
): SecurityFinding[] {
  const findings: SecurityFinding[] = [];

  try {
    const fileContent = content ?? (fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf-8') : '');
    if (!fileContent) return findings;

    const lineOffsets = buildLineOffsets(fileContent);

    for (const { pattern, rule, severity, message, suggestion } of patterns) {
      const regex = new RegExp(pattern.source, pattern.flags);
      let match;
      while ((match = regex.exec(fileContent)) !== null) {
        const lineNum = lineNumberAt(lineOffsets, match.index);
        findings.push({
          file: filePath,
          line: lineNum,
          severity,
          rule,
          message,
          match: match[0].slice(0, 50),
          suggestion,
        });
      }
    }
  } catch {
    // Skip unreadable files
  }

  return findings;
}

/**
 * Scan multiple files for security issues
 */
export function scanFiles(
  files: string[],
  patterns: SecurityPattern[] = SECURITY_PATTERNS,
  maxFiles: number = 100
): SecurityFinding[] {
  const findings: SecurityFinding[] = [];

  for (const file of files.slice(0, maxFiles)) {
    findings.push(...scanFile(file, undefined, patterns));
  }

  return findings;
}

/**
 * Get severity score (for sorting/filtering)
 */
export function getSeverityScore(severity: string): number {
  switch (severity) {
    case 'critical': return 4;
    case 'high': return 3;
    case 'medium': return 2;
    case 'low': return 1;
    default: return 0;
  }
}

/**
 * Sort findings by severity (highest first)
 */
export function sortBySeverity(findings: SecurityFinding[]): SecurityFinding[] {
  return [...findings].sort((a, b) => getSeverityScore(b.severity) - getSeverityScore(a.severity));
}

export default {
  SECURITY_PATTERNS,
  scanFile,
  scanFiles,
  getSeverityScore,
  sortBySeverity,
};
