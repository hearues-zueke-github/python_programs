//! Author: Hearues Zueke
//! E-Mail: hziko314@gmail.com
//! Datetime: 2026.01.22T19:12+1

const std = @import("std");
const utils = @import("utils.zig");

const stdout = utils.stdout;
const printU8ArrayAsHex = utils.printU8ArrayAsHex;

pub const HashSha256PRNG = struct {
    const sha256: type = std.crypto.hash.sha2.Sha256;

    allocator: std.mem.Allocator,
    l_bytearr_seed: [][]u8,
    l_state: [][]u8,
    i_state: usize,
    amount_state: usize,
    l_t_i_state: [][]usize,
    buffer_hash: [32]u8,

    pub fn init(allocator: std.mem.Allocator, l_bytearr: []const []const u8) !HashSha256PRNG {
        const amount_state: usize = l_bytearr.len;
        var l_bytearr_seed: [][]u8 = try allocator.alloc([]u8, amount_state);
        var l_state: [][]u8 = try allocator.alloc([]u8, amount_state);
        var l_t_i_state: [][]usize = try allocator.alloc([]usize, amount_state);
        
        {
            var i: usize = 0;
            while (i < amount_state) : (i += 1) {
                l_bytearr_seed[i] = try allocator.dupe(u8, l_bytearr[i]);
            }
        }

        l_t_i_state[0] = try allocator.dupe(usize, &.{amount_state-1, 0, 1});
        if (amount_state > 1) {
            l_t_i_state[amount_state-1] = try allocator.dupe(usize, &.{amount_state-2, amount_state-1, 0});
        }
        
        {
            var i: usize = 1;
            while (i < amount_state-1) : (i += 1) {
                l_t_i_state[i] = try allocator.dupe(usize, &.{i-1, i, i+1});
            }
        }

        {
            var i: usize = 0;
            while (i < amount_state) : (i += 1) {
                l_state[i] = try allocator.alloc(u8, 32);
                sha256.hash(l_bytearr[i], &(l_state[i][0..32].*), .{});
            }
        }

        var self: HashSha256PRNG = HashSha256PRNG{
            .l_bytearr_seed=l_bytearr_seed,
            .l_state=l_state,
            .i_state=0,
            .amount_state=amount_state,
            .l_t_i_state=l_t_i_state,
            .allocator=allocator,
            .buffer_hash=[_]u8{0} ** 32,
        };

        try stdout.print("amount_state: {d}\n", .{self.amount_state});

        try stdout.print("- round_i: {}\n", .{0});
        try self.printState();

        for (0..(self.amount_state*2)) |round_i| {
            try stdout.print("- round_i: {}\n", .{round_i+1});
            self.calc_next_state_and_next_random_bytearray();
            try self.printState();
            try stdout.flush();
        }

        return self;
    }

    pub fn deinit(self: HashSha256PRNG) void {
        const allocator: std.mem.Allocator = self.allocator;

        var i: u32 = 0;
        while (i < self.amount_state) : (i += 1) {
            allocator.destroy(self.l_bytearr_seed[i]);
        }
        allocator.destroy(self.l_bytearr_seed);

        i = 0;
        while (i < self.amount_state) : (i += 1) {
            allocator.destroy(self.l_state[i]);
        }
        allocator.destroy(self.l_state);

        i = 0;
        while (i < self.amount_state) : (i += 1) {
            allocator.destroy(self.l_t_i_state[i]);
        }
        allocator.destroy(self.l_t_i_state);
    }

    fn calc_next_state_and_next_random_bytearray(self: *HashSha256PRNG) void {
        const t_i_state: []usize = self.l_t_i_state[self.i_state];
        const i_state_prev = t_i_state[0];
        const i_state_curr = t_i_state[1];
        const i_state_next = t_i_state[2];

        const state_prev: []u8 = self.l_state[i_state_prev];
        var state_curr: []u8 = self.l_state[i_state_curr];
        const state_next: []u8 = self.l_state[i_state_next];

        sha256.hash(state_curr, &self.buffer_hash, .{});

        var state_prev_hash: [32]u8 = [_]u8{0} ** 32;
        sha256.hash(state_prev, &state_prev_hash, .{});

        for (0..32) |i| {
            state_curr[i] = (state_prev_hash[i] ^ state_curr[i]) +% state_next[i];
        }

        self.i_state = (self.i_state + 1) % self.amount_state;
    }

    pub fn printState(self: HashSha256PRNG) !void {
        for (self.l_state, 0..) |state, i| {
            try stdout.print("-- l_state[{d}]: state: ", .{i});
            try self.printOneState(state);
            try stdout.print("\n", .{});
        }
    }

    fn printOneState(_: HashSha256PRNG, state: []u8) !void {
        try stdout.print("{X:0>2}", .{state[0]});
        for (state[1..]) |value| {
            try stdout.print(",{X:0>2}", .{value});
        }
    }

    pub fn generate_random_u8_values(self: *HashSha256PRNG, arr: *[]u8) !void {
        const full_rounds: usize = arr.*.len / 32;
        const rest_bytes = arr.*.len % 32;

        var pos_i: usize = 0;
        for (0..full_rounds) |_| {
            self.calc_next_state_and_next_random_bytearray();
            try stdout.print("buffer_hash: ", .{});
            try printU8ArrayAsHex(&self.buffer_hash);
            try stdout.print("\n", .{});
            try stdout.flush();

            for (0..32) |i| {
                arr.*[pos_i] = self.buffer_hash[i];
                pos_i += 1;
            }
        }

        if (rest_bytes > 0) {
            self.calc_next_state_and_next_random_bytearray();
            try stdout.print("buffer_hash: ", .{});
            try printU8ArrayAsHex(&self.buffer_hash);
            try stdout.print("\n", .{});
            try stdout.flush();

            for (0..rest_bytes) |i| {
                arr.*[pos_i] = self.buffer_hash[i];
                pos_i += 1;
            }
        }
    }

    pub fn generate_random_u32_values(self: *HashSha256PRNG, arr_u32: *[]u32) !void {
        var arr_u8: []u8 = std.mem.sliceAsBytes(arr_u32.*);
        try self.generate_random_u8_values(&arr_u8);
    }

    pub fn generate_random_u8_modulo_values(self: *HashSha256PRNG, arr_u8: *[]u8, modulo: u8) !void {
        try self.generate_random_u8_values(arr_u8);
        for (0..arr_u8.len) |i| {
            arr_u8.*[i] %= modulo;
        }
    }
};
