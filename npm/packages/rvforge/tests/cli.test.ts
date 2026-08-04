import { ForgeError, ForgeErrorCode } from '../src/errors';
import { parseArgs, resolveMode, splitBuildOperands } from '../src/cli';

describe('parseArgs', () => {
  it('separates the command, positionals, and flags', () => {
    const args = parseArgs(['build', 'agent.rvf', 'linux', '--out', 'dist', '--json']);
    expect(args.command).toBe('build');
    expect(args.positionals).toEqual(['agent.rvf', 'linux']);
    expect(args.flags.get('out')).toBe('dist');
    expect(args.flags.get('json')).toBe(true);
  });

  it('accepts --flag=value form', () => {
    expect(parseArgs(['build', '--out=dist']).flags.get('out')).toBe('dist');
  });

  it('collects repeated --only flags', () => {
    const args = parseArgs(['download', 'b1', '--only', 'a.exe', '--only', 'b.msi']);
    expect(args.repeated.get('only')).toEqual(['a.exe', 'b.msi']);
  });

  it('expands -h and -v', () => {
    expect(parseArgs(['-h']).flags.get('help')).toBe(true);
    expect(parseArgs(['-v']).flags.get('version')).toBe(true);
  });

  it('rejects a value flag with no value', () => {
    expect(() => parseArgs(['build', '--out'])).toThrow(ForgeError);
    expect(() => parseArgs(['build', '--out', '--json'])).toThrow(ForgeError);
  });

  it('rejects a value on a boolean flag', () => {
    try {
      parseArgs(['build', '--json=yes']);
      throw new Error('expected a ForgeError');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.USAGE);
    }
  });

  it('treats everything after -- as positional', () => {
    expect(parseArgs(['verify', '--', '--weird-name.exe']).positionals).toEqual(['--weird-name.exe']);
  });
});

describe('splitBuildOperands', () => {
  it('reads no operands as "use the config"', () => {
    expect(splitBuildOperands([])).toEqual({});
  });

  it('reads a lone .rvf as the payload', () => {
    expect(splitBuildOperands(['agent.rvf'])).toEqual({ rvfPath: 'agent.rvf' });
  });

  it('reads a lone non-.rvf operand as a target, not a path', () => {
    expect(splitBuildOperands(['linux-x64'])).toEqual({ targets: ['linux-x64'] });
  });

  it('reads a path followed by targets', () => {
    expect(splitBuildOperands(['dir/agent.RVF', 'linux', 'windows'])).toEqual({
      rvfPath: 'dir/agent.RVF',
      targets: ['linux', 'windows'],
    });
  });

  it('reads several targets with no path', () => {
    expect(splitBuildOperands(['linux', 'macos'])).toEqual({ targets: ['linux', 'macos'] });
  });
});

describe('resolveMode', () => {
  it('leaves the configured mode in place when --mode is absent', () => {
    expect(resolveMode(undefined)).toBeUndefined();
  });

  it('accepts every packaging mode, case- and space-insensitively', () => {
    expect(resolveMode('embedded')).toBe('embedded');
    expect(resolveMode('  Thin ')).toBe('thin');
    expect(resolveMode('SHARED-READER')).toBe('shared-reader');
  });

  it('rejects an unknown mode as a usage error', () => {
    try {
      resolveMode('capsule');
      throw new Error('expected a ForgeError');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.USAGE);
      expect((err as ForgeError).message).toMatch(/embedded/);
    }
  });

  it('parses --mode as a value flag', () => {
    expect(parseArgs(['build', '--mode', 'thin']).flags.get('mode')).toBe('thin');
    expect(parseArgs(['build', '--mode=embedded']).flags.get('mode')).toBe('embedded');
  });
});
