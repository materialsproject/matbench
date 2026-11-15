import inflect
p = inflect.engine()

def num_to_words(n):
    def float_to_words(f):
        if f >= 1:
            return f"{p.number_to_words(int(f))} point {' '.join([p.number_to_words(int(x)) for x in str(round(f - int(f), 2))[2:]])}"
        else:
            decimal_str = ('%.15f' % f).split('.')[1]
            decimal_str = decimal_str.rstrip('0')
            decimal_len = len(decimal_str)
            
            if decimal_len == 0:
                decimal_str = '000'
            elif decimal_len <= 3:
                decimal_str = decimal_str.ljust(3, '0')
                return f"{p.number_to_words(int(decimal_str))} thousandths"
            elif decimal_len <= 6:
                decimal_str = decimal_str.ljust(6, '0')
                return f"{p.number_to_words(int(decimal_str))} millionths"
            else:
                decimal_str = decimal_str.ljust(9, '0')
                return f"{p.number_to_words(int(decimal_str))} billionths"

    if isinstance(n, int):
        return p.number_to_words(n)
    elif isinstance(n, float):
        return float_to_words(n)
    else:
        return "Input is not a valid number"

if __name__ == "__main__":
    number = float(input("请输入一个数字: "))
    words = num_to_words(number)
    print(f"{number} 转换为英文单词是: {words}")
