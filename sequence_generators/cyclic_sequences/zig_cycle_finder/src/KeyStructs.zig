//! Author: Hearues Zueke
//! E-Mail: hziko314@gmail.com
//! Datetime: 2026.01.22T19:12+1

const std = @import("std");

pub const KeyModuloXPower = struct {
    modulo: u32,
    x_val: u32,
    power: u32,
};

pub const KeyModuloXPowerContext = struct {
    const Self = @This();

    pub fn hash(_: Self, key: *KeyModuloXPower) u64 {
        var h = std.hash.Wyhash.init(0);  // <- change the hash algo according to your needs... (WyHash...)
        h.update(std.mem.asBytes(&key.modulo));
        h.update(std.mem.asBytes(&key.x_val));
        h.update(std.mem.asBytes(&key.power));
        return h.final();
    }

    pub fn eql(_: Self, a: *KeyModuloXPower, b: *KeyModuloXPower) bool {
        return (a.modulo == b.modulo) and (a.x_val == b.x_val) and (a.power == b.power);
    }
};

pub const KeyFactorPowers = struct {
    factor: u32,
    powers: []const u32,
};

pub const KeyFactorPowersContext = struct {
    const Self = @This();
    ascending: bool = true,

    pub fn hash(_: Self, key: *KeyFactorPowers) u64 {
        var h = std.hash.Wyhash.init(0);  // <- change the hash algo according to your needs... (WyHash...)
        h.update(std.mem.asBytes(&key.factor));
        h.update(std.mem.sliceAsBytes(key.powers));
        return h.final();
    }

    pub fn eql(_: Self, a: *KeyFactorPowers, b: *KeyFactorPowers) bool {
        return (a.factor == b.factor) and std.mem.eql(u32, a.powers, b.powers);
    }

    pub fn lt(self: Self, a: *KeyFactorPowers, b: *KeyFactorPowers) bool {
        if (self.ascending) {
            return (
                (a.*.factor < b.*.factor) or
                (
                    (a.*.factor == b.*.factor) and
                    (std.mem.order(u32, a.*.powers, b.*.powers) == .lt)
                )
            );
        } else {
            return (
                (a.*.factor > b.*.factor) or
                (
                    (a.*.factor == b.*.factor) and
                    (std.mem.order(u32, a.*.powers, b.*.powers) == .gt)
                )
            );
        }
    }

    pub fn lt_order(self: Self, a: *KeyFactorPowers, b: *KeyFactorPowers) std.mem.Order {
        if (self.ascending) {
            if (a.*.factor < b.*.factor) {
                return .lt;
            } else if (a.*.factor == b.*.factor) {
                if (std.mem.order(u32, a.*.powers, b.*.powers) == .lt) {
                    return .lt;
                } else if (std.mem.order(u32, a.*.powers, b.*.powers) == .eq) {
                    return .eq;
                }

                return .gt;
            }

            return .gt;
        } else {
            if (a.*.factor > b.*.factor) {
                return .gt;
            } else if (a.*.factor == b.*.factor) {
                if (std.mem.order(u32, a.*.powers, b.*.powers) == .gt) {
                    return .gt;
                } else if (std.mem.order(u32, a.*.powers, b.*.powers) == .eq) {
                    return .eq;
                }

                return .lt;
            }

            return .lt;
        }
    }     
};

pub const KeyXVals = struct {
    x_vals: []const u32,
};

pub const KeyXValsContext = struct {
    const Self = @This();

    pub fn hash(_: Self, key: *KeyXVals) u64 {
        var h = std.hash.Wyhash.init(0);  // <- change the hash algo according to your needs... (WyHash...)
        h.update(std.mem.sliceAsBytes(key.x_vals));
        return h.final();
    }

    pub fn eql(_: Self, a: *KeyXVals, b: *KeyXVals) bool {
        return std.mem.eql(u32, a.x_vals, b.x_vals);
    }
};

pub const KeyFactorXValsPowers = struct {
    factor: u32,
    x_vals: []const u32,
    powers: []const u32,
};

pub const KeyFactorXValsPowersContext = struct {
    const Self = @This();
    
    pub fn hash(_: Self, key: *KeyFactorXValsPowers) u64 {
        var h = std.hash.Wyhash.init(0);  // <- change the hash algo according to your needs... (WyHash...)
        h.update(std.mem.asBytes(&key.factor));
        h.update(std.mem.sliceAsBytes(key.x_vals));
        h.update(std.mem.sliceAsBytes(key.powers));
        return h.final();
    }

    pub fn eql(_: Self, a: *KeyFactorXValsPowers, b: *KeyFactorXValsPowers) bool {
        return (
            (a.factor == b.factor) and
            std.mem.eql(u32, a.x_vals, b.x_vals) and
            std.mem.eql(u32, a.powers, b.powers)
        );
    }
};
