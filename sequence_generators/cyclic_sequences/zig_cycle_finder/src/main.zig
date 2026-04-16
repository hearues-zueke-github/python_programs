//! Author: Hearues Zueke
//! E-Mail: hziko314@gmail.com
//! Datetime: 2026.01.22T19:12+1

const std = @import("std");
const zig_cycle_finder = @import("zig_cycle_finder");
const expect = std.testing.expect;
const HashSha256PRNG = @import("HashSha256PRNG.zig").HashSha256PRNG;

const utils = @import("utils.zig");

const stdout = utils.stdout;
const printU8ArrayAsHex = utils.printU8ArrayAsHex;
const printU8ArrayAsDec = utils.printU8ArrayAsDec;
const printU32ArrayAsHex = utils.printU32ArrayAsHex;

const KeyStructs = @import("KeyStructs.zig");

const KeyModuloXPower = KeyStructs.KeyModuloXPower;
const KeyModuloXPowerContext = KeyStructs.KeyModuloXPowerContext;
const KeyFactorPowers = KeyStructs.KeyFactorPowers;
const KeyFactorPowersContext = KeyStructs.KeyFactorPowersContext;
const KeyXVals = KeyStructs.KeyXVals;
const KeyXValsContext = KeyStructs.KeyXValsContext;
const KeyFactorXValsPowers = KeyStructs.KeyFactorXValsPowers;
const KeyFactorXValsPowersContext = KeyStructs.KeyFactorXValsPowersContext;

const HashMapKeyModuloXPowerToU32 = std.HashMap(*KeyModuloXPower, u32, KeyModuloXPowerContext, std.hash_map.default_max_load_percentage);
const HashMapKeyXValsToU32 = std.HashMap(*KeyXVals, u32, KeyXValsContext, std.hash_map.default_max_load_percentage);
const HashMapKeyFactorPowersToHashMapKeyXValsToU32 = std.HashMap(*KeyFactorPowers, HashMapKeyXValsToU32, KeyFactorPowersContext, std.hash_map.default_max_load_percentage);
const HashMapKeyFactorXValsPowersToU32 = std.HashMap(*KeyFactorXValsPowers, u32, KeyFactorXValsPowersContext, std.hash_map.default_max_load_percentage);

// Helper to compare slices of T using a custom order function
fn sliceOrder(comptime T: type, lhs: []const T, rhs: []const T, cmp: fn (T, T) std.math.Order) std.math.Order {
    const len = @min(lhs.len, rhs.len);
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const result = cmp(lhs[i], rhs[i]);
        if (result != .eq) return result;
    }
    return std.math.order(lhs.len, rhs.len);
}

const OneCounter = struct {
    const Self = @This();

    count: u32,
    count_start: u32,
    count_max: u32,

    fn count_next(self: *Self) bool {
        self.count += 1;

        if (self.count > self.count_max) {
            self.count = self.count_start;
            return true;
        }

        return false;
    }
};

const Counter = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    arr_one_counter: []*OneCounter,

    fn init(allocator: std.mem.Allocator, arr_count_start: []const u32, arr_count_max: []const u32) !Self {
        if (arr_count_start.len != arr_count_max.len) {
            return error.Unexpected;
        }

        const amount_counter: usize = arr_count_start.len;
        var arr_one_counter: []*OneCounter = try allocator.alloc(*OneCounter, amount_counter);

        for (0..amount_counter) |i| {
            const one_counter: *OneCounter = try allocator.create(OneCounter);
            arr_one_counter[i] = one_counter;
            one_counter.* = .{
                .count=arr_count_start[i],
                .count_start=arr_count_start[i],
                .count_max=arr_count_max[i],
            };
        }

        const self: Self = Self{
            .allocator=allocator,
            .arr_one_counter=arr_one_counter,
        };
        return self;
    }

    fn deinit(self: *Self) void {
        for (0..self.arr_one_counter.len) |i| {
            self.allocator.destroy(self.arr_one_counter[i]);
        }
        self.allocator.free(self.arr_one_counter);
    }

    fn count_next(self: *Self) void {
        for (0..self.arr_one_counter.len) |i| {
            if (!self.arr_one_counter[i].count_next()) {
                break;
            }
        }
    }

    fn print_count(self: Self) !void {
        try stdout.print("arr_counter: [", .{});
        for (0..self.arr_one_counter.len) |i| {
            try stdout.print("{d}, ", .{self.arr_one_counter[i].*.count});
        }
        try stdout.print("]", .{});
        try stdout.flush();
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator: std.mem.Allocator = gpa.allocator();

    var hash_sha256_prng: HashSha256PRNG = try HashSha256PRNG.init(allocator, &.{
            "",
            "1",
            "12",
            "125",
            "129",
        }
    );
    
    var arr: []u8 = try allocator.alloc(u8, 40);
    defer allocator.free(arr);

    try hash_sha256_prng.generate_random_u8_values(&arr);

    try stdout.print("arr: ", .{});
    try printU8ArrayAsHex(arr);
    try stdout.print("\n", .{});
    try stdout.flush();


    var arr_u32: []u32 = try allocator.alloc(u32, 11);
    defer allocator.free(arr_u32);

    try hash_sha256_prng.generate_random_u32_values(&arr_u32);

    try stdout.print("arr_u32: ", .{});
    try printU32ArrayAsHex(arr_u32);
    try stdout.print("\n", .{});
    try stdout.flush();


    var arr_u8_modulo: []u8 = try allocator.alloc(u8, 11);
    defer allocator.free(arr_u8_modulo);

    try hash_sha256_prng.generate_random_u8_modulo_values(&arr_u8_modulo, 23);

    try stdout.print("arr_u8_modulo: ", .{});
    try printU8ArrayAsDec(arr_u8_modulo);
    try stdout.print("\n", .{});
    try stdout.flush();

    // try hash_sha256_prng.printState();

    var map_modulo_x_power_to_u32: HashMapKeyModuloXPowerToU32 = .init(allocator);
    defer {
        var iter = map_modulo_x_power_to_u32.iterator();
        while (iter.next()) |kv| {
            allocator.destroy(kv.key_ptr.*);
        }
        map_modulo_x_power_to_u32.deinit();
    }

    const modulo: u32 = 8;

    // x = 0 and power = 0
    {
        const key_modulo_x_power: *KeyModuloXPower = try allocator.create(KeyModuloXPower);
        key_modulo_x_power.* = .{.modulo=modulo, .x_val=0, .power=0};

        try map_modulo_x_power_to_u32.put(key_modulo_x_power, 1);
    }

    // x = 0 and power = 1..modulo
    {
        var power: u32 = 1;
        while (power < modulo) : (power += 1) {
            const key_modulo_x_power: *KeyModuloXPower = try allocator.create(KeyModuloXPower);
            key_modulo_x_power.* = .{.modulo=modulo, .x_val=0, .power=power};

            try map_modulo_x_power_to_u32.put(key_modulo_x_power, 0);
        }
    }

    // x = 1..modulo and power = 1..power
    {
        var x_val: u32 = 1;
        while (x_val < modulo) : (x_val += 1) {
            var val_mod: u32 = 1;

            var power: u32 = 0;
            while (power < modulo) : (power += 1) {
                const key_modulo_x_power: *KeyModuloXPower = try allocator.create(KeyModuloXPower);
                key_modulo_x_power.* = .{.modulo=modulo, .x_val=x_val, .power=power};

                try map_modulo_x_power_to_u32.put(key_modulo_x_power, val_mod);

                val_mod = (val_mod * x_val) % modulo;
            }
        }
    }


    var map_factor_powers_to_map: HashMapKeyFactorPowersToHashMapKeyXValsToU32 = .init(allocator);
    defer {
        var iter = map_factor_powers_to_map.iterator();
        while (iter.next()) |kv| {
            allocator.free(kv.key_ptr.*.powers);
            allocator.destroy(kv.key_ptr.*);
            
            var iter2 = kv.value_ptr.iterator();
            while (iter2.next()) |kv2| {
                allocator.free(kv2.key_ptr.*.x_vals);
                allocator.destroy(kv2.key_ptr.*);
            }

            kv.value_ptr.deinit();
        }

        map_factor_powers_to_map.deinit();
    }

    var elapsed_sec_hashmap_key_factor_powers: f64 = 0.0;

    {
        var timer = try std.time.Timer.start();

        var factor: u32 = 1;
        while (factor < modulo) : (factor += 1) {
            var power_0: u32 = 0;
            while (power_0 < modulo) : (power_0 += 1) {
                var power_1: u32 = 0;
                while (power_1 < modulo) : (power_1 += 1) {
                    const key_factor_powers: *KeyFactorPowers = try allocator.create(KeyFactorPowers);
                    key_factor_powers.* = .{.factor=factor, .powers=try allocator.dupe(u32, &.{power_0, power_1})};

                    try map_factor_powers_to_map.put(key_factor_powers, .init(allocator));
                    const map_x_vals: *HashMapKeyXValsToU32 = map_factor_powers_to_map.getPtr(key_factor_powers).?;

                    var x_val_0: u32 = 0;
                    while (x_val_0 < modulo) : (x_val_0 += 1) {
                        var x_val_1: u32 = 0;
                        while (x_val_1 < modulo) : (x_val_1 += 1) {
                            const key_x_vals: *KeyXVals = try allocator.create(KeyXVals);
                            key_x_vals.* = .{.x_vals=try allocator.dupe(u32, &.{x_val_0, x_val_1})};
                            
                            var val_mod: u32 = factor;
                            var key_x_val_0: KeyModuloXPower = KeyModuloXPower{.modulo=modulo, .x_val=x_val_0, .power=power_0};
                            var key_x_val_1: KeyModuloXPower = KeyModuloXPower{.modulo=modulo, .x_val=x_val_1, .power=power_1};
                            
                            val_mod = (val_mod * map_modulo_x_power_to_u32.get(&key_x_val_0).?) % modulo;
                            val_mod = (val_mod * map_modulo_x_power_to_u32.get(&key_x_val_1).?) % modulo;

                            try map_x_vals.put(key_x_vals, val_mod);
                        }
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        elapsed_sec_hashmap_key_factor_powers = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    }

    var map_factor_x_vals_powers_to_map: HashMapKeyFactorXValsPowersToU32 = .init(allocator);
    defer {
        var iter = map_factor_x_vals_powers_to_map.iterator();
        while (iter.next()) |kv| {
            allocator.free(kv.key_ptr.*.x_vals);
            allocator.free(kv.key_ptr.*.powers);
            allocator.destroy(kv.key_ptr.*);
        }

        map_factor_x_vals_powers_to_map.deinit();
    }

    var elapsed_sec_hashmap_key_factor_x_vals_powers: f64 = 0.0;

    {
        var timer = try std.time.Timer.start();

        var factor: u32 = 1;
        while (factor < modulo) : (factor += 1) {
            
            var power_0: u32 = 0;
            while (power_0 < modulo) : (power_0 += 1) {
                var x_val_0: u32 = 0;
                while (x_val_0 < modulo) : (x_val_0 += 1) {
                    var key_x_val_0: KeyModuloXPower = KeyModuloXPower{.modulo=modulo, .x_val=x_val_0, .power=power_0};

                    var power_1: u32 = 0;
                    while (power_1 < modulo) : (power_1 += 1) {
                        var x_val_1: u32 = 0;
                        while (x_val_1 < modulo) : (x_val_1 += 1) {
                            var key_x_val_1: KeyModuloXPower = KeyModuloXPower{.modulo=modulo, .x_val=x_val_1, .power=power_1};
            
                            const key_factor_x_vals_powers: *KeyFactorXValsPowers = try allocator.create(KeyFactorXValsPowers);
                            key_factor_x_vals_powers.* = .{
                                .factor=factor, 
                                .x_vals=try allocator.dupe(u32, &.{x_val_0, x_val_1}),
                                .powers=try allocator.dupe(u32, &.{power_0, power_1}),
                            };

                            var val_mod: u32 = factor;
                            val_mod = (val_mod * map_modulo_x_power_to_u32.get(&key_x_val_0).?) % modulo;
                            val_mod = (val_mod * map_modulo_x_power_to_u32.get(&key_x_val_1).?) % modulo;

                            try map_factor_x_vals_powers_to_map.put(key_factor_x_vals_powers, val_mod);
                        }
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        elapsed_sec_hashmap_key_factor_x_vals_powers = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    }

    try stdout.print("modulo: {d}\n", .{modulo});
    try stdout.print("elapsed_sec_hashmap_key_factor_powers: {d} s\n", .{elapsed_sec_hashmap_key_factor_powers});
    try stdout.print("elapsed_sec_hashmap_key_factor_x_vals_powers: {d} s\n", .{elapsed_sec_hashmap_key_factor_x_vals_powers});
    
    // IMPORTANT: You must flush at the end to actually send data to the screen
    try stdout.flush();

    // // std.debug.print("current map_factor_powers_to_map:\n", .{});
    // var iter = map_factor_powers_to_map.iterator();
    // while (iter.next()) |kv1| {
    //     // std.debug.print("- key_factor_powers: {any}, value map size: {d}\n", .{kv1.key_ptr.*, kv1.value_ptr.*.count()});
    //     var iter2 = kv1.value_ptr.*.iterator();
    //     while (iter2.next()) |kv2| {
    //         // std.debug.print("-- key_x_vals: {any}, val_mod: {d}\n", .{kv2.key_ptr.*, kv2.value_ptr.*});
    //         std.debug.print("{d}, {d}, {d}, {d}, {d}, {d}\n", .{
    //             kv1.key_ptr.*.factor,
    //             kv1.key_ptr.*.powers[0],
    //             kv1.key_ptr.*.powers[1],
    //             kv2.key_ptr.*.x_vals[0],
    //             kv2.key_ptr.*.x_vals[1],
    //             kv2.value_ptr.*,
    //         });
    //     }
    // }

    // var counter: Counter = try .init(
    //     allocator,
    //     &.{
    //         1, 0, 0,
    //         1, 0, 0,
    //         1, 0, 0,
    //         1, 0, 0,
    //     }, &.{
    //         modulo-1, modulo-1, modulo-1,
    //         modulo-1, modulo-1, modulo-1,
    //         modulo-1, modulo-1, modulo-1,
    //         modulo-1, modulo-1, modulo-1,
    //     },
    // );
    // defer counter.deinit();

    // try stdout.print("idx_counter: {d}, counter: ", .{0});
    // try counter.print_count();
    // try stdout.print("\n", .{});
    // try stdout.flush();

    // for (1..150) |idx_counter| {
    //     counter.count_next();
        
    //     try stdout.print("idx_counter: {d}, counter: ", .{idx_counter});
    //     try counter.print_count();
    //     try stdout.print("\n", .{});
    //     try stdout.flush();
    // }

    // // 2. Initialize the ArrayList
    // var list = std.ArrayList(u32).init(allocator);
    // // 3. Always defer deinit to prevent memory leaks
    // defer list.deinit();

    // // 4. Append items (these can fail, so use 'try')
    // try list.append(10);
    // try list.append(20);
    // try list.appendSlice(&.{ 30, 40, 50 });

    // // 5. Accessing and Modifying items
    // list.items[0] = 100;

    // // 6. Iterating over the list
    // for (list.items, 0..) |val, i| {
    //     std.debug.print("Index {d}: {d}\n", .{ i, val });
    // }

    const HashMapCycleLenToCount = std.AutoHashMap(u32, u64);
    // const HashMapCycleLenToArrayListFactors = std.HashMap(u32, std.ArrayList([]u32));
    var map_cycle_len_to_count: HashMapCycleLenToCount = .init(allocator);
    defer {
        map_cycle_len_to_count.deinit();
    }


    var keys_sorted: std.ArrayList(*KeyFactorPowers) = .empty;
    defer keys_sorted.deinit(allocator);
    var keys_sorted_constant: std.ArrayList(*KeyFactorPowers) = .empty;
    defer keys_sorted_constant.deinit(allocator);
    
    {   
        var it = map_factor_powers_to_map.keyIterator();
        while (it.next()) |key_ptr| {
            if (key_ptr.*.powers[0] == 0 and key_ptr.*.powers[1] == 0) {
                try keys_sorted_constant.append(allocator, key_ptr.*);
            } else {
                try keys_sorted.append(allocator, key_ptr.*);
            }
        }
    }

    // 2. Sort the ArrayList
    // For strings, use a custom lessThan function
    std.mem.sort(*KeyFactorPowers, keys_sorted.items, {}, struct {
        fn lessThan(_: void, a: *KeyFactorPowers, b: *KeyFactorPowers) bool {
            if (a.*.factor < b.*.factor) {
                return true;
            } else if (a.*.factor == b.*.factor) {
                return (std.mem.order(u32, a.powers, b.powers) == .lt);
            }

            return false;
        }
    }.lessThan);

    // // 3. Print sorted keys_sorted
    // for (keys_sorted.items) |key| {
    //     std.debug.print("{s}: {d}\n", .{ key, map.get(key).? });
    // }


    var arr_idx_visited: []u32 = try allocator.alloc(u32, modulo*modulo);
    var arr_value: []u8 = try allocator.alloc(u8, modulo*modulo*2);
    defer allocator.free(arr_idx_visited);
    defer allocator.free(arr_value);

    const TupleArrayPtrKeyFactorPowers = struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        arr_polynome_factor_powers_0: []*KeyFactorPowers,
        arr_polynome_factor_powers_1: []*KeyFactorPowers,

        fn init(_allocator: std.mem.Allocator, arr_polynome_factor_powers_0: []*KeyFactorPowers, arr_polynome_factor_powers_1: []*KeyFactorPowers) !Self {
            var arr_polynome_factor_powers_0_dupe = try allocator.dupe(*KeyFactorPowers, arr_polynome_factor_powers_0);
            var arr_polynome_factor_powers_1_dupe = try allocator.dupe(*KeyFactorPowers, arr_polynome_factor_powers_1);

            std.mem.sort([]*KeyFactorPowers, &arr_polynome_factor_powers_0_dupe, {}, KeyFactorPowersContext.lt);
            std.mem.sort([]*KeyFactorPowers, &arr_polynome_factor_powers_1_dupe, {}, KeyFactorPowersContext.lt);

            const self = Self{
                .allocator=_allocator,
                .arr_polynome_factor_powers_0=arr_polynome_factor_powers_0_dupe,
                .arr_polynome_factor_powers_1=arr_polynome_factor_powers_1_dupe,
            };

            return self;
        }

        fn deinit(self: *Self) void {
            self.allocator.free(self.arr_polynome_factor_powers_0);
            self.allocator.free(self.arr_polynome_factor_powers_1);
        }
    };

    const TupleArrayPtrKeyFactorPowersContext = struct {
        const Self = @This();

        pub fn hash(_: Self, key: *TupleArrayPtrKeyFactorPowers) u64 {
            var h = std.hash.Wyhash.init(0);  // <- change the hash algo according to your needs... (WyHash...)
            for (key.*.arr_polynome_factor_powers_0) |polynome_factor_powers_0| {
                for (polynome_factor_powers_0) |factor_powers| {
                    h.update(std.mem.asBytes(&factor_powers.modulo));
                    h.update(std.mem.sliceAsBytes(&factor_powers.powers));
                }
            }
            return h.final();
        }

        pub fn eql(_: Self, a: *TupleArrayPtrKeyFactorPowers, b: *TupleArrayPtrKeyFactorPowers) bool {
            return sliceOrder(
                *KeyFactorPowers,
                a.*.arr_polynome_factor_powers_0,
                b.*.arr_polynome_factor_powers_0,
                KeyFactorPowersContext.lt_order,
            ) == .eq;
        }
    };

    var found_max_cycle: u32 = 0;

    // var arrlist_tpl_array_ptr_key_factor_powers: std.ArrayList(*TupleArrayPtrKeyFactorPowers) = .empty;
    var arrlist_tpl_array_ptr_key_factor_powers: std.Hashmap(*TupleArrayPtrKeyFactorPowers) = .empty;
    defer {
        for (arrlist_tpl_array_ptr_key_factor_powers.items) |ptr| {
            allocator.destroy(ptr);
        }
        arrlist_tpl_array_ptr_key_factor_powers.deinit(allocator);
    }

    const HashMapTupleArrayPtrKeyFactorPowersToBool = std.HashMap(*TupleArrayPtrKeyFactorPowers, bool, TupleArrayPtrKeyFactorPowersContext, std.hash_map.default_max_load_percentage);
    var map_tuple_array_ptr_key_factor_powers_to_bool: HashMapTupleArrayPtrKeyFactorPowersToBool = .init(allocator);
    defer {
        var iter = map_tuple_array_ptr_key_factor_powers_to_bool.iterator();
        while (iter.next()) |kv| {
            allocator.destroy(kv.key_ptr.*);
        }
        map_tuple_array_ptr_key_factor_powers_to_bool.deinit();
    }

    var idx_counter: usize = 0;
    // const modulo_fac_full_cycle: usize = (modulo-1)*modulo*modulo;
    // const max_count: usize = modulo_fac_full_cycle * modulo_fac_full_cycle * modulo_fac_full_cycle;
    // const max_count: usize = modulo_fac_full_cycle * modulo_fac_full_cycle * modulo_fac_full_cycle * modulo_fac_full_cycle;
    // for (0..max_count) |_| {
    for (0..keys_sorted.items.len) |idx_key_0| {
    for (idx_key_0+1..keys_sorted.items.len) |idx_key_1| {
    for (idx_key_1+1..keys_sorted.items.len) |idx_key_2| {
    for (0..keys_sorted_constant.items.len) |idx_key_const_0| {
    for (0..keys_sorted.items.len) |idx_key_3| {
    for (0..keys_sorted_constant.items.len) |idx_key_const_1| {
    // for (idx_key_2+1..keys_sorted.items.len) |idx_key_3| {
        idx_counter += 1;
        // counter.count_next();

        if (idx_counter % 1000000 == 0) {
            try stdout.print("idx_counter: {d}\n", .{idx_counter});
            try stdout.flush();

            // try stdout.print("idx_counter: {d}, counter: ", .{idx_counter});
            // try counter.print_count();
            // try stdout.print("\n", .{});
            // try stdout.flush();

            try stdout.print("Print of map_cycle_len_to_count:\n", .{});
            try stdout.print("{{", .{});

            var iterator = map_cycle_len_to_count.iterator();
            while (iterator.next()) |entry| {
                try stdout.print("({d}, {d}), ", .{entry.key_ptr.*, entry.value_ptr.*});
            }
            
            try stdout.print("}}\n\n", .{});
            try stdout.flush();
        }

        // try stdout.print("idx_counter: {d}, counter: ", .{idx_counter});
        // try counter.print_count();
        // try stdout.print("\n", .{});
        // try stdout.flush();



        // test with some simple values
        const arr_polynome_factor_powers_0: [4]*KeyFactorPowers = .{
            // .{.factor=counter.arr_one_counter[0].count, .powers=&.{counter.arr_one_counter[1].count, counter.arr_one_counter[2].count}},
            // .{.factor=counter.arr_one_counter[3].count, .powers=&.{counter.arr_one_counter[4].count, counter.arr_one_counter[5].count}},
            // .{.factor=counter.arr_one_counter[6].count, .powers=&.{counter.arr_one_counter[7].count, counter.arr_one_counter[8].count}},
            // .{.factor=counter.arr_one_counter[9].count, .powers=&.{counter.arr_one_counter[10].count, counter.arr_one_counter[11].count}},
            keys_sorted.items[idx_key_0],
            keys_sorted.items[idx_key_1],
            keys_sorted.items[idx_key_2],
            // keys_sorted.items[idx_key_2].*,
            keys_sorted_constant.items[idx_key_const_0],
            // keys_sorted.items[idx_key_3].*,
            // .{.factor=@as(u32, @intCast(i_factor_1)), .powers=&.{2, 3}},
            // .{.factor=2, .powers=&.{1, 5}},
            // .{.factor=4, .powers=&.{2, 1}},
        };

        const arr_polynome_factor_powers_1: [2]*KeyFactorPowers = .{
            keys_sorted.items[idx_key_3],
            keys_sorted_constant.items[idx_key_const_1],
            // .{.factor=1, .powers=&.{1, 0}},
        };
        
        // try stdout.print("- arr_polynome_factor_powers_0:\n", .{});
        // try stdout.flush();
        // for (0..arr_polynome_factor_powers_0.len) |i| {
        //     const key_factor_powers: *const KeyFactorPowers = &arr_polynome_factor_powers_0[i];
        //     try stdout.print("-- i: {d}, key_factor_powers: {any}\n", .{i, key_factor_powers});
        //     try stdout.flush();
        // }

        // try stdout.print("- arr_polynome_factor_powers_1:\n", .{});
        // try stdout.flush();
        // for (0..arr_polynome_factor_powers_1.len) |i| {
        //     const key_factor_powers: *const KeyFactorPowers = &arr_polynome_factor_powers_1[i];
        //     try stdout.print("-- i: {d}, key_factor_powers: {any}\n", .{i, key_factor_powers});
        //     try stdout.flush();
        // }

        @memset(arr_idx_visited, 0);
        @memset(arr_value, 0);

        // try stdout.print("-before:\n", .{});
        // try stdout.print("-- arr_idx_visited: {any}\n", .{arr_idx_visited});
        // try stdout.print("-- arr_value: {any}\n", .{arr_value});
        // try stdout.flush();

        var x_0_curr: u32 = 0;
        var x_1_curr: u32 = 0;

        const constant_0: u32 = 1;
        const constant_1: u32 = 1;
        var cycle_len: u32 = 0;

        arr_idx_visited[0] = 1;
        for (0..modulo*modulo) |i| {
            const key_x_vals: KeyXVals = .{.x_vals=&.{x_0_curr, x_1_curr}};

            var x_0_next: u32 = 0;
            for (arr_polynome_factor_powers_0) |key_factor_powers| {
                const map_x_vals: *HashMapKeyXValsToU32 = map_factor_powers_to_map.getPtr(@constCast(key_factor_powers)).?;

                const mod_val: u32 = map_x_vals.get(@constCast(&key_x_vals)).?;
                x_0_next = (x_0_next + mod_val) % modulo;
            }
            x_0_next = (x_0_next + constant_0) % modulo;

            var x_1_next: u32 = 0;
            for (arr_polynome_factor_powers_1) |key_factor_powers| {
                const map_x_vals: *HashMapKeyXValsToU32 = map_factor_powers_to_map.getPtr(@constCast(key_factor_powers)).?;

                const mod_val: u32 = map_x_vals.get(@constCast(&key_x_vals)).?;
                x_1_next = (x_1_next + mod_val) % modulo;
            }
            x_1_next = (x_1_next + constant_1) % modulo;

            // try stdout.print("--- x_0_next: {}, x_1_next: {}\n", .{x_0_next, x_1_next});
            try stdout.flush();

            const idx_pos_next: u32 = x_0_next * modulo + x_1_next;
            const idx_count: u32 = arr_idx_visited[idx_pos_next];
            if (idx_count != 0) {
                cycle_len = @as(u32, @intCast(i + 2)) - idx_count;
                break;
            }

            arr_idx_visited[idx_pos_next] = @as(u32, @intCast(i + 2));

            arr_value[2*(i+1)+0] = @truncate(x_0_next);
            arr_value[2*(i+1)+1] = @truncate(x_1_next);

            x_0_curr = x_0_next;
            x_1_curr = x_1_next;
        }

        // // try stdout.print("- after:\n", .{});
        // try stdout.print("-- arr_idx_visited: {any}\n", .{arr_idx_visited});
        // try stdout.print("-- arr_value: {any}\n", .{arr_value});
        // try stdout.print("-- cycle_len: {d}\n", .{cycle_len});
        // try stdout.flush();

        if (!map_cycle_len_to_count.contains(cycle_len)) {
            try map_cycle_len_to_count.put(cycle_len, 0);
        }

        map_cycle_len_to_count.getPtr(cycle_len).?.* += 1;

        if (found_max_cycle < cycle_len) {
            found_max_cycle = cycle_len;

            for (arrlist_tpl_array_ptr_key_factor_powers.items) |ptr| {
                allocator.destroy(ptr);
            }
            arrlist_tpl_array_ptr_key_factor_powers.deinit(allocator);
        }
    }
    }
    }
    }
    }
    }

    try stdout.print("idx_counter: {d}\n", .{idx_counter});
    try stdout.flush();
    
    // try stdout.print("idx_counter: {d}, counter: ", .{idx_counter});
    // try counter.print_count();
    // try stdout.print("\n", .{});
    // try stdout.flush();

    try stdout.print("Print of map_cycle_len_to_count:\n", .{});
    try stdout.print("{{", .{});

    var iterator = map_cycle_len_to_count.iterator();
    while (iterator.next()) |entry| {
        try stdout.print("({d}, {d}), ", .{entry.key_ptr.*, entry.value_ptr.*});
    }
    
    try stdout.print("}}\n", .{});
    try stdout.flush();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
